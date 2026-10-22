#अन्तर्भाव "tiled.h"
#अन्तर्भाव "tiled-kernel.h"

#अन्तर्भाव "ggml-cpu-impl.h"
#अन्तर्भाव "ggml-cpu.h"
#अन्तर्भाव "ggml.h"

// kvalues table (impl section) for the iq4_xs unpack
#परिभाषय GGML_COMMON_IMPL_CPP
#अन्तर्भाव "ggml-common.h"

#अन्तर्भाव <cassert>
#अन्तर्भाव <cstdlib>
#अन्तर्भाव <cstring>

#अन्तर्भाव <mutex>

#परिभाषय अप्रयुक्तम् GGML_अप्रयुक्तम्

// unpack routines for various quant types src0
स्थिर शून्य tiled_unpack_src0(ध्रुव खण्डम्_q4_K * पङ्क्तिकाः, int64_t row_stride, पूर्णाङ्क n_पङ्क्तिकाः, tiled_tile_src0 * tile, पूर्णाङ्क अङ्कः_k) {
    GGML_ASSERT(n_पङ्क्तिकाः <= TILED_TILE_पङ्क्तिकाः);
    constexpr पूर्णाङ्क NB = 8; // 32-wide subblocks
    // 12-byte packed scale/min decode, same extraction as the reference kernels
    स्थिर ध्रुव uint32_t kmask1 = 0x3f3f3f3f;
    स्थिर ध्रुव uint32_t kmask2 = 0x0f0f0f0f;
    स्थिर ध्रुव uint32_t kmask3 = 0x03030303;

    ध्रुव पूर्णाङ्क qk_stride = अङ्कः_k * TILED_TILE_K;
    ध्रुव पूर्णाङ्क nb_stride = NB * अङ्कः_k;
    प्रति (पूर्णाङ्क slab = 0; slab < अङ्कः_k; slab++) {
        प्रति (पूर्णाङ्क r = 0; r < n_पङ्क्तिकाः; r++) {
            ध्रुव पूर्णाङ्क d_off = slab * TILED_MICRO + r;
            ध्रुव पूर्णाङ्क q_off = r * qk_stride + slab * TILED_TILE_K;
            ध्रुव पूर्णाङ्क s_off = r * nb_stride + slab * NB;
            ध्रुव खण्डम्_q4_K & x = पङ्क्तिकाः[r * row_stride + slab];

            tile->d[d_off]    = ggml_fp16_गन्तव्यम्_fp32(x.दत्त.दत्त.d);
            tile->dmin[d_off] = ggml_fp16_गन्तव्यम्_fp32(x.दत्त.दत्त.dmin);

            uint32_t utmp[4];
            स्मृतिं_प्रतिलिख(utmp, x.scales, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            ध्रुव uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;

            ध्रुव uint8_t * scales = (ध्रुव uint8_t *) &utmp[0];
            ध्रुव uint8_t * mins = (ध्रुव uint8_t *) &utmp[2];
            प्रति (पूर्णाङ्क s = 0; s < NB; s++) {
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

स्थिर शून्य tiled_unpack_src0(ध्रुव खण्डम्_q5_K * पङ्क्तिकाः, int64_t row_stride, पूर्णाङ्क n_पङ्क्तिकाः, tiled_tile_src0 * tile, पूर्णाङ्क अङ्कः_k) {
    GGML_ASSERT(n_पङ्क्तिकाः <= TILED_TILE_पङ्क्तिकाः);
    constexpr पूर्णाङ्क NB = 8; // 32-wide subblocks
    // 12-byte packed scale/min decode, same extraction as the reference kernels
    स्थिर ध्रुव uint32_t kmask1 = 0x3f3f3f3f;
    स्थिर ध्रुव uint32_t kmask2 = 0x0f0f0f0f;
    स्थिर ध्रुव uint32_t kmask3 = 0x03030303;

    ध्रुव पूर्णाङ्क qk_stride = अङ्कः_k * TILED_TILE_K;
    ध्रुव पूर्णाङ्क nb_stride = NB * अङ्कः_k;
    प्रति (पूर्णाङ्क slab = 0; slab < अङ्कः_k; slab++) {
        प्रति (पूर्णाङ्क r = 0; r < n_पङ्क्तिकाः; r++) {
            ध्रुव पूर्णाङ्क d_off = slab * TILED_MICRO + r;
            ध्रुव पूर्णाङ्क q_off = r * qk_stride + slab * TILED_TILE_K;
            ध्रुव पूर्णाङ्क s_off = r * nb_stride + slab * NB;
            ध्रुव खण्डम्_q5_K & x = पङ्क्तिकाः[r * row_stride + slab];

            tile->d[d_off]    = ggml_fp16_गन्तव्यम्_fp32(x.दत्त.दत्त.d);
            tile->dmin[d_off] = ggml_fp16_गन्तव्यम्_fp32(x.दत्त.दत्त.dmin);

            uint32_t utmp[4];
            स्मृतिं_प्रतिलिख(utmp, x.scales, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            ध्रुव uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;

            ध्रुव uint8_t * scales = (ध्रुव uint8_t *) &utmp[0];
            ध्रुव uint8_t * mins = (ध्रुव uint8_t *) &utmp[2];
            प्रति (पूर्णाङ्क s = 0; s < NB; s++) {
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

स्थिर शून्य tiled_unpack_src0(ध्रुव खण्डम्_q6_K * पङ्क्तिकाः, int64_t row_stride, पूर्णाङ्क n_पङ्क्तिकाः, tiled_tile_src0 * tile, पूर्णाङ्क अङ्कः_k) {
    GGML_ASSERT(n_पङ्क्तिकाः <= TILED_TILE_पङ्क्तिकाः);
    constexpr पूर्णाङ्क NB = 16; // 16-wide subblocks
    ध्रुव पूर्णाङ्क qk_stride = अङ्कः_k * TILED_TILE_K;
    ध्रुव पूर्णाङ्क nb_stride = NB * अङ्कः_k;
    प्रति (पूर्णाङ्क slab = 0; slab < अङ्कः_k; slab++) {
        प्रति (पूर्णाङ्क r = 0; r < n_पङ्क्तिकाः; r++) {
            ध्रुव पूर्णाङ्क d_off = slab * TILED_MICRO + r;
            ध्रुव पूर्णाङ्क q_off = r * qk_stride + slab * TILED_TILE_K;
            ध्रुव पूर्णाङ्क s_off = r * nb_stride + slab * NB;
            ध्रुव खण्डम्_q6_K & x = पङ्क्तिकाः[r * row_stride + slab];
            tile->d[d_off] = ggml_fp16_गन्तव्यम्_fp32(x.d);

            // 6-bit code = 4 low bits (ql) | 2 high bits (qh); see ggml_vec_dot_q6_K_q8_K_generic
            // per half the lanes are [ql lo(0:32)] [ql lo(32:64)] [ql hi(0:32)] [ql hi(32:64)]
            uint8_t * q = &tile->q[q_off];
            प्रति (पूर्णाङ्क half = 0; half < 2; half++) {
                uint8_t * बहिर्गमनम् = q + 128 * half;
                tiled_unpk_nib4(x.ql + 64 * half + 0,  बहिर्गमनम् + 0,  बहिर्गमनम् + 64);
                tiled_unpk_nib4(x.ql + 64 * half + 32, बहिर्गमनम् + 32, बहिर्गमनम् + 96);
                tiled_unpk_or<0, 4, 3>(बहिर्गमनम् + 0,  x.qh + 32 * half);
                tiled_unpk_or<2, 4, 3>(बहिर्गमनम् + 32, x.qh + 32 * half);
                tiled_unpk_or<4, 4, 3>(बहिर्गमनम् + 64, x.qh + 32 * half);
                tiled_unpk_or<6, 4, 3>(बहिर्गमनम् + 96, x.qh + 32 * half);
            }
            // scale is a plain int8 per 16-element subblock (16 per 256-K)
            प्रति (पूर्णाङ्क s = 0; s < NB; s++) { tile->scales[s_off + s] = (int32_t) (int8_t) x.scales[s]; }
        }
    }
}

स्थिर शून्य tiled_unpack_src0(ध्रुव खण्डम्_q3_K * पङ्क्तिकाः, int64_t row_stride, पूर्णाङ्क n_पङ्क्तिकाः, tiled_tile_src0 * tile, पूर्णाङ्क अङ्कः_k) {
    GGML_ASSERT(n_पङ्क्तिकाः <= TILED_TILE_पङ्क्तिकाः);
    constexpr पूर्णाङ्क NB = 16; // 16-wide subblocks
    ध्रुव पूर्णाङ्क qk_stride = अङ्कः_k * TILED_TILE_K;
    ध्रुव पूर्णाङ्क nb_stride = NB * अङ्कः_k;
    प्रति (पूर्णाङ्क slab = 0; slab < अङ्कः_k; slab++) {
        प्रति (पूर्णाङ्क r = 0; r < n_पङ्क्तिकाः; r++) {
            ध्रुव पूर्णाङ्क d_off = slab * TILED_MICRO + r;
            ध्रुव पूर्णाङ्क q_off = r * qk_stride + slab * TILED_TILE_K;
            ध्रुव पूर्णाङ्क s_off = r * nb_stride + slab * NB;
            ध्रुव खण्डम्_q3_K & x = पङ्क्तिकाः[r * row_stride + slab];
            tile->d[d_off] = ggml_fp16_गन्तव्यम्_fp32(x.d);

            // 3-bit code = 2 low bits (qs) | (1 high bit from hmask << 2)
            // element e (0..255): half=e>>7, el=e&127, group=el>>5, l=el&31
            //   low2 = (qs[half*32 + l] >> 2*group) & 3
            //   high = (hmask[l] >> (half*4 + group)) & 1
            // see ggml_vec_dot_q3_K_q8_K_generic
            uint8_t * q = &tile->q[q_off];
            ध्रुव uint8_t * s0 = x.qs;
            ध्रुव uint8_t * s1 = x.qs + 32;
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
            स्थिर ध्रुव uint32_t kmask1 = 0x03030303;
            स्थिर ध्रुव uint32_t kmask2 = 0x0f0f0f0f;
            uint32_t auxs[4];
            स्मृतिं_प्रतिलिख(auxs, x.scales, 12);
            ध्रुव uint32_t tmp = auxs[2];
            auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            ध्रुव int8_t * scales = (ध्रुव int8_t *) &auxs[0];
            प्रति (पूर्णाङ्क s = 0; s < NB; s++) { tile->scales[s_off + s] = (int32_t) scales[s] - 32; }
        }
    }
}

स्थिर शून्य tiled_unpack_src0(ध्रुव खण्डम्_q2_K * पङ्क्तिकाः, int64_t row_stride, पूर्णाङ्क n_पङ्क्तिकाः, tiled_tile_src0 * tile, पूर्णाङ्क अङ्कः_k) {
    GGML_ASSERT(n_पङ्क्तिकाः <= TILED_TILE_पङ्क्तिकाः);
    constexpr पूर्णाङ्क NB = 16; // 16-wide subblocks
    ध्रुव पूर्णाङ्क qk_stride = अङ्कः_k * TILED_TILE_K;
    ध्रुव पूर्णाङ्क nb_stride = NB * अङ्कः_k;
    प्रति (पूर्णाङ्क slab = 0; slab < अङ्कः_k; slab++) {
        प्रति (पूर्णाङ्क r = 0; r < n_पङ्क्तिकाः; r++) {
            ध्रुव पूर्णाङ्क d_off = slab * TILED_MICRO + r;
            ध्रुव पूर्णाङ्क q_off = r * qk_stride + slab * TILED_TILE_K;
            ध्रुव पूर्णाङ्क s_off = r * nb_stride + slab * NB;
            ध्रुव खण्डम्_q2_K & x = पङ्क्तिकाः[r * row_stride + slab];
            tile->d[d_off]    = ggml_fp16_गन्तव्यम्_fp32(x.दत्त.दत्त.d);
            tile->dmin[d_off] = ggml_fp16_गन्तव्यम्_fp32(x.दत्त.दत्त.dmin);

            // 2-bit code: element e -> half=e>>7, el=e&127
            //   byte = half*32 + (el & 31), shift = 2*(el >> 5)
            // see ggml_vec_dot_q2_K_q8_K_generic
            uint8_t * q = &tile->q[q_off];
            प्रति (पूर्णाङ्क half = 0; half < 2; half++) {
                ध्रुव uint8_t * s = x.qs + 32 * half;
                uint8_t * बहिर्गमनम् = q + 128 * half;
                tiled_unpk_2bit<0>(s, बहिर्गमनम् + 0);
                tiled_unpk_2bit<2>(s, बहिर्गमनम् + 32);
                tiled_unpk_2bit<4>(s, बहिर्गमनम् + 64);
                tiled_unpk_2bit<6>(s, बहिर्गमनम् + 96);
            }
            // scale/min packed in one byte per 16-element subblock: low 4 bits = scale, high 4 = min
            प्रति (पूर्णाङ्क s = 0; s < NB; s++) {
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
स्थिर शून्य tiled_unpack_src0(ध्रुव खण्डम्_iq4_xs * पङ्क्तिकाः, int64_t row_stride, पूर्णाङ्क n_पङ्क्तिकाः, tiled_tile_src0 * tile, पूर्णाङ्क अङ्कः_k) {
    GGML_ASSERT(n_पङ्क्तिकाः <= TILED_TILE_पङ्क्तिकाः);
    constexpr पूर्णाङ्क NB = 8; // 32-wide subblocks

    स्थिर ध्रुव uint8_t lut[16] = {
        (uint8_t) (kvalues_iq4nl[0] + 128),  (uint8_t) (kvalues_iq4nl[1] + 128),
        (uint8_t) (kvalues_iq4nl[2] + 128),  (uint8_t) (kvalues_iq4nl[3] + 128),
        (uint8_t) (kvalues_iq4nl[4] + 128),  (uint8_t) (kvalues_iq4nl[5] + 128),
        (uint8_t) (kvalues_iq4nl[6] + 128),  (uint8_t) (kvalues_iq4nl[7] + 128),
        (uint8_t) (kvalues_iq4nl[8] + 128),  (uint8_t) (kvalues_iq4nl[9] + 128),
        (uint8_t) (kvalues_iq4nl[10] + 128), (uint8_t) (kvalues_iq4nl[11] + 128),
        (uint8_t) (kvalues_iq4nl[12] + 128), (uint8_t) (kvalues_iq4nl[13] + 128),
        (uint8_t) (kvalues_iq4nl[14] + 128), (uint8_t) (kvalues_iq4nl[15] + 128),
    };

    ध्रुव पूर्णाङ्क qk_stride = अङ्कः_k * TILED_TILE_K;
    ध्रुव पूर्णाङ्क nb_stride = NB * अङ्कः_k;
    प्रति (पूर्णाङ्क slab = 0; slab < अङ्कः_k; slab++) {
        प्रति (पूर्णाङ्क r = 0; r < n_पङ्क्तिकाः; r++) {
            ध्रुव पूर्णाङ्क d_off = slab * TILED_MICRO + r;
            ध्रुव पूर्णाङ्क q_off = r * qk_stride + slab * TILED_TILE_K;
            ध्रुव पूर्णाङ्क s_off = r * nb_stride + slab * NB;
            ध्रुव खण्डम्_iq4_xs & x = पङ्क्तिकाः[r * row_stride + slab];
            tile->d[d_off] = ggml_fp16_गन्तव्यम्_fp32(x.d);

            // 6-bit scale per 32, stored as (ls - 32); same extraction as dequantize_row_iq4_xs
            प्रति (पूर्णाङ्क s = 0; s < NB; s++) {
                ध्रुव पूर्णाङ्क ls = ((x.scales_l[s / 2] >> 4 * (s % 2)) & 0xf) | (((x.scales_h >> 2 * s) & 3) << 4);
                tile->scales[s_off + s] = (int32_t) ls - 32;
            }

            // 4-bit codes through the LUT: low nibbles of a byte pair come first
            uint8_t * q = &tile->q[q_off];
            uint8_t lo[32], hi[32];
            प्रति (पूर्णाङ्क u = 0; u < 4; u++) {
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
स्थिर शून्य tiled_unpack_src0(ध्रुव खण्डम्_iq2_xxs * पङ्क्तिकाः, int64_t row_stride, पूर्णाङ्क n_पङ्क्तिकाः, tiled_tile_src0 * tile, पूर्णाङ्क अङ्कः_k) {
    GGML_ASSERT(n_पङ्क्तिकाः <= TILED_TILE_पङ्क्तिकाः);
    constexpr पूर्णाङ्क NB = 8; // 32-wide subblocks

    ध्रुव पूर्णाङ्क qk_stride = अङ्कः_k * TILED_TILE_K;
    ध्रुव पूर्णाङ्क nb_stride = NB * अङ्कः_k;
    प्रति (पूर्णाङ्क slab = 0; slab < अङ्कः_k; slab++) {
        प्रति (पूर्णाङ्क r = 0; r < n_पङ्क्तिकाः; r++) {
            ध्रुव पूर्णाङ्क d_off = slab * TILED_MICRO + r;
            ध्रुव पूर्णाङ्क q_off = r * qk_stride + slab * TILED_TILE_K;
            ध्रुव पूर्णाङ्क s_off = r * nb_stride + slab * NB;
            ध्रुव खण्डम्_iq2_xxs & x = पङ्क्तिकाः[r * row_stride + slab];
            tile->d[d_off] = ggml_fp16_गन्तव्यम्_fp32(x.d) * 0.125f;

            uint32_t aux32[2];
            ध्रुव uint8_t * aux8 = (ध्रुव uint8_t *) aux32;
            uint64_t g[4];
            uint8_t signs4[4];
            प्रति (पूर्णाङ्क ib32 = 0; ib32 < NB; ib32++) {
                स्मृतिं_प्रतिलिख(aux32, x.qs + 4 * ib32, 2 * परिमाण(uint32_t));
                tile->scales[s_off + ib32] = (int32_t) (2 * (aux32[1] >> 28) + 1);
                प्रति (पूर्णाङ्क l = 0; l < 4; l++) {
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
स्थिर शून्य tiled_unpack_src0(ध्रुव खण्डम्_iq2_xs * पङ्क्तिकाः, int64_t row_stride, पूर्णाङ्क n_पङ्क्तिकाः, tiled_tile_src0 * tile, पूर्णाङ्क अङ्कः_k) {
    GGML_ASSERT(n_पङ्क्तिकाः <= TILED_TILE_पङ्क्तिकाः);
    constexpr पूर्णाङ्क NB = 16; // 16-wide subblocks

    ध्रुव पूर्णाङ्क qk_stride = अङ्कः_k * TILED_TILE_K;
    ध्रुव पूर्णाङ्क nb_stride = NB * अङ्कः_k;
    प्रति (पूर्णाङ्क slab = 0; slab < अङ्कः_k; slab++) {
        प्रति (पूर्णाङ्क r = 0; r < n_पङ्क्तिकाः; r++) {
            ध्रुव पूर्णाङ्क d_off = slab * TILED_MICRO + r;
            ध्रुव पूर्णाङ्क q_off = r * qk_stride + slab * TILED_TILE_K;
            ध्रुव पूर्णाङ्क s_off = r * nb_stride + slab * NB;
            ध्रुव खण्डम्_iq2_xs & x = पङ्क्तिकाः[r * row_stride + slab];
            tile->d[d_off] = ggml_fp16_गन्तव्यम्_fp32(x.d) * 0.125f;

            uint64_t g[4];
            uint8_t signs4[4];
            प्रति (पूर्णाङ्क ib32 = 0; ib32 < QK_K / 32; ib32++) {
                tile->scales[s_off + 2 * ib32 + 0] = (int32_t) (2 * (x.scales[ib32] & 0xf) + 1);
                tile->scales[s_off + 2 * ib32 + 1] = (int32_t) (2 * (x.scales[ib32] >> 4) + 1);
                ध्रुव uint16_t * q = x.qs + 4 * ib32;
                प्रति (पूर्णाङ्क l = 0; l < 4; l++) {
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
स्थिर शून्य tiled_unpack_src0(ध्रुव खण्डम्_iq2_s * पङ्क्तिकाः, int64_t row_stride, पूर्णाङ्क n_पङ्क्तिकाः, tiled_tile_src0 * tile, पूर्णाङ्क अङ्कः_k) {
    GGML_ASSERT(n_पङ्क्तिकाः <= TILED_TILE_पङ्क्तिकाः);
    constexpr पूर्णाङ्क NB = 16; // 16-wide subblocks

    ध्रुव पूर्णाङ्क qk_stride = अङ्कः_k * TILED_TILE_K;
    ध्रुव पूर्णाङ्क nb_stride = NB * अङ्कः_k;
    प्रति (पूर्णाङ्क slab = 0; slab < अङ्कः_k; slab++) {
        प्रति (पूर्णाङ्क r = 0; r < n_पङ्क्तिकाः; r++) {
            ध्रुव पूर्णाङ्क d_off = slab * TILED_MICRO + r;
            ध्रुव पूर्णाङ्क q_off = r * qk_stride + slab * TILED_TILE_K;
            ध्रुव पूर्णाङ्क s_off = r * nb_stride + slab * NB;
            ध्रुव खण्डम्_iq2_s & x = पङ्क्तिकाः[r * row_stride + slab];
            tile->d[d_off] = ggml_fp16_गन्तव्यम्_fp32(x.d) * 0.125f;

            ध्रुव uint8_t * qs = x.qs;
            ध्रुव uint8_t * signs = x.qs + QK_K / 8; // packed sign bytes share the qs array, same as the reference
            uint64_t g[4];
            प्रति (पूर्णाङ्क ib32 = 0; ib32 < QK_K / 32; ib32++) {
                tile->scales[s_off + 2 * ib32 + 0] = (int32_t) (2 * (x.scales[ib32] & 0xf) + 1);
                tile->scales[s_off + 2 * ib32 + 1] = (int32_t) (2 * (x.scales[ib32] >> 4) + 1);
                प्रति (पूर्णाङ्क l = 0; l < 4; l++) {
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
स्थिर शून्य tiled_unpack_src0(ध्रुव खण्डम्_iq3_xxs * पङ्क्तिकाः, int64_t row_stride, पूर्णाङ्क n_पङ्क्तिकाः, tiled_tile_src0 * tile, पूर्णाङ्क अङ्कः_k) {
    GGML_ASSERT(n_पङ्क्तिकाः <= TILED_TILE_पङ्क्तिकाः);
    constexpr पूर्णाङ्क NB = 8; // 32-wide subblocks

    ध्रुव पूर्णाङ्क qk_stride = अङ्कः_k * TILED_TILE_K;
    ध्रुव पूर्णाङ्क nb_stride = NB * अङ्कः_k;
    प्रति (पूर्णाङ्क slab = 0; slab < अङ्कः_k; slab++) {
        प्रति (पूर्णाङ्क r = 0; r < n_पङ्क्तिकाः; r++) {
            ध्रुव पूर्णाङ्क d_off = slab * TILED_MICRO + r;
            ध्रुव पूर्णाङ्क q_off = r * qk_stride + slab * TILED_TILE_K;
            ध्रुव पूर्णाङ्क s_off = r * nb_stride + slab * NB;
            ध्रुव खण्डम्_iq3_xxs & x = पङ्क्तिकाः[r * row_stride + slab];
            tile->d[d_off] = ggml_fp16_गन्तव्यम्_fp32(x.d) * 0.25f;

            ध्रुव uint8_t * qs = x.qs;
            ध्रुव uint8_t * scales_and_signs = x.qs + QK_K / 4; // 4 bytes per 32: code bits in the top nibble, signs in 7-bit chunks
            uint64_t g[4];
            uint8_t signs4[4];
            प्रति (पूर्णाङ्क ib32 = 0; ib32 < QK_K / 32; ib32++) {
                ध्रुव uint32_t aux32 = *(ध्रुव uint32_t *) (scales_and_signs + 4 * ib32);
                tile->scales[s_off + ib32] = (int32_t) (2 * (aux32 >> 28) + 1);
                प्रति (पूर्णाङ्क l = 0; l < 4; l++) {
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
स्थिर शून्य tiled_unpack_src0(ध्रुव खण्डम्_iq3_s * पङ्क्तिकाः, int64_t row_stride, पूर्णाङ्क n_पङ्क्तिकाः, tiled_tile_src0 * tile, पूर्णाङ्क अङ्कः_k) {
    GGML_ASSERT(n_पङ्क्तिकाः <= TILED_TILE_पङ्क्तिकाः);
    constexpr पूर्णाङ्क NB = 8; // 32-wide subblocks

    ध्रुव पूर्णाङ्क qk_stride = अङ्कः_k * TILED_TILE_K;
    ध्रुव पूर्णाङ्क nb_stride = NB * अङ्कः_k;
    प्रति (पूर्णाङ्क slab = 0; slab < अङ्कः_k; slab++) {
        प्रति (पूर्णाङ्क r = 0; r < n_पङ्क्तिकाः; r++) {
            ध्रुव पूर्णाङ्क d_off = slab * TILED_MICRO + r;
            ध्रुव पूर्णाङ्क q_off = r * qk_stride + slab * TILED_TILE_K;
            ध्रुव पूर्णाङ्क s_off = r * nb_stride + slab * NB;
            ध्रुव खण्डम्_iq3_s & x = पङ्क्तिकाः[r * row_stride + slab];
            tile->d[d_off] = ggml_fp16_गन्तव्यम्_fp32(x.d);

            ध्रुव uint8_t * qs = x.qs;
            ध्रुव uint8_t * qh = x.qh;
            ध्रुव uint8_t * signs = x.signs;
            प्रति (पूर्णाङ्क ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
                tile->scales[s_off + ib32 + 0] = (int32_t) (1 + 2 * (x.scales[ib32 / 2] & 0xf));
                tile->scales[s_off + ib32 + 1] = (int32_t) (1 + 2 * (x.scales[ib32 / 2] >> 4));
                प्रति (पूर्णाङ्क h = 0; h < 2; h++) {
                    uint64_t g[4];
                    प्रति (पूर्णाङ्क l = 0; l < 4; l++) {
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
स्थिर शून्य tiled_unpack_src0(ध्रुव खण्डम्_iq1_s * पङ्क्तिकाः, int64_t row_stride, पूर्णाङ्क n_पङ्क्तिकाः, tiled_tile_src0 * tile, पूर्णाङ्क अङ्कः_k) {
    GGML_ASSERT(n_पङ्क्तिकाः <= TILED_TILE_पङ्क्तिकाः);
    constexpr पूर्णाङ्क NB = 8; // 32-wide subblocks

    ध्रुव पूर्णाङ्क qk_stride = अङ्कः_k * TILED_TILE_K;
    ध्रुव पूर्णाङ्क nb_stride = NB * अङ्कः_k;
    प्रति (पूर्णाङ्क slab = 0; slab < अङ्कः_k; slab++) {
        प्रति (पूर्णाङ्क r = 0; r < n_पङ्क्तिकाः; r++) {
            ध्रुव पूर्णाङ्क d_off = slab * TILED_MICRO + r;
            ध्रुव पूर्णाङ्क q_off = r * qk_stride + slab * TILED_TILE_K;
            ध्रुव पूर्णाङ्क s_off = r * nb_stride + slab * NB;
            ध्रुव खण्डम्_iq1_s & x = पङ्क्तिकाः[r * row_stride + slab];
            tile->d[d_off] = ggml_fp16_गन्तव्यम्_fp32(x.d) * 0.125f;

            ध्रुव uint8_t * qs = x.qs;
            प्रति (पूर्णाङ्क ib = 0; ib < QK_K / 32; ib++) {
                ध्रुव uint16_t hw = x.qh[ib];
                tile->scales[s_off + ib] = (int32_t) (2 * ((hw >> 12) & 7) + 1);
                ध्रुव int8_t delta = (hw & 0x8000) ? -1 : 1;
                प्रति (पूर्णाङ्क l = 0; l < 4; l++) {
                    ध्रुव uint64_t प्रविष्टिः = iq1s_grid[qs[l] | (((hw >> (3 * l)) & 7) << 8)];
                    tiled_unpk_tern8((ध्रुव uint8_t *) &प्रविष्टिः, delta, &tile->q[q_off + 32 * ib + 8 * l]);
                }
                qs += 4;
            }
        }
    }
}

// iq1_m: like iq1_s but the fp16 scale is packed across the 4 scale bytes (no d field) and the
// delta offset is per 8, giving one scale per 16; 3-bit scale per 16, no min
// codes stored as (8 * grid + delta + 128), matching BIAS = 128
स्थिर शून्य tiled_unpack_src0(ध्रुव खण्डम्_iq1_m * पङ्क्तिकाः, int64_t row_stride, पूर्णाङ्क n_पङ्क्तिकाः, tiled_tile_src0 * tile, पूर्णाङ्क अङ्कः_k) {
    GGML_ASSERT(n_पङ्क्तिकाः <= TILED_TILE_पङ्क्तिकाः);
    constexpr पूर्णाङ्क NB = 16; // 16-wide subblocks

    ध्रुव पूर्णाङ्क qk_stride = अङ्कः_k * TILED_TILE_K;
    ध्रुव पूर्णाङ्क nb_stride = NB * अङ्कः_k;
    प्रति (पूर्णाङ्क slab = 0; slab < अङ्कः_k; slab++) {
        प्रति (पूर्णाङ्क r = 0; r < n_पङ्क्तिकाः; r++) {
            ध्रुव पूर्णाङ्क d_off = slab * TILED_MICRO + r;
            ध्रुव पूर्णाङ्क q_off = r * qk_stride + slab * TILED_TILE_K;
            ध्रुव पूर्णाङ्क s_off = r * nb_stride + slab * NB;
            ध्रुव खण्डम्_iq1_m & x = पङ्क्तिकाः[r * row_stride + slab];

            ध्रुव uint16_t * sc = (ध्रुव uint16_t *) x.scales;
            iq1m_scale_t scale;
            scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
            tile->d[d_off] = ggml_fp16_गन्तव्यम्_fp32(scale.f16) * 0.125f;

            ध्रुव uint8_t * qs = x.qs;
            ध्रुव uint8_t * qh = x.qh;
            प्रति (पूर्णाङ्क ib = 0; ib < QK_K / 32; ib++) {
                ध्रुव uint16_t hw = sc[ib / 2];
                ध्रुव पूर्णाङ्क sh = 6 * (ib % 2);
                tile->scales[s_off + 2 * ib + 0] = (int32_t) (2 * ((hw >> sh) & 7) + 1);
                tile->scales[s_off + 2 * ib + 1] = (int32_t) (2 * ((hw >> (sh + 3)) & 7) + 1);

                ध्रुव uint16_t idx[4] = {
                    (uint16_t) (qs[0] | ((qh[0] << 8) & 0x700)),
                    (uint16_t) (qs[1] | ((qh[0] << 4) & 0x700)),
                    (uint16_t) (qs[2] | ((qh[1] << 8) & 0x700)),
                    (uint16_t) (qs[3] | ((qh[1] << 4) & 0x700)),
                };
                ध्रुव int8_t delta[4] = {
                    (int8_t) ((qh[0] & 0x08) ? -1 : 1), (int8_t) ((qh[0] & 0x80) ? -1 : 1),
                    (int8_t) ((qh[1] & 0x08) ? -1 : 1), (int8_t) ((qh[1] & 0x80) ? -1 : 1),
                };
                प्रति (पूर्णाङ्क l = 0; l < 4; l++) {
                    ध्रुव uint64_t प्रविष्टिः = iq1s_grid[idx[l]];
                    tiled_unpk_tern8((ध्रुव uint8_t *) &प्रविष्टिः, delta[l], &tile->q[q_off + 32 * ib + 8 * l]);
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
स्थिर शून्य tiled_unpack_src1_q8_K(ध्रुव खण्डम्_q8_K * ध्रुव * पङ्क्तिकाः, पूर्णाङ्क n_पङ्क्तिकाः, tiled_tile_src1 * tile,
                                   पूर्णाङ्क kblk, पूर्णाङ्क अङ्कः_k) {
    GGML_ASSERT(n_पङ्क्तिकाः <= TILED_TILE_पङ्क्तिकाः);
    ध्रुव पूर्णाङ्क n_padded = (n_पङ्क्तिकाः + TILED_MICRO - 1) & ~(TILED_MICRO - 1);
    ध्रुव पूर्णाङ्क bs_stride = (अङ्कः_k == 1) ? TILED_TILE_पङ्क्तिकाः : TILED_MICRO;
    ध्रुव पूर्णाङ्क n16 = TILED_TILE_K / TILED_MICRO;
    // natural [row][k] fill, zero-pad ragged tail; codes stay natural here, the driver
    // repacks them just-in-time via tiled_repack_src1 (per 16-row band on the standard path,
    // the whole chunk on the narrow path) so each repacked region is L1-hot for its uses
    प्रति (पूर्णाङ्क slab = 0; slab < अङ्कः_k; slab++) {
        ध्रुव पूर्णाङ्क q_off = slab * TILED_TILE_K;
        प्रति (पूर्णाङ्क r = 0; r < n_padded; r++) {
            यदि (r < n_पङ्क्तिकाः) {
                स्मृतिं_प्रतिलिख(&tile->q[r * (अङ्कः_k * TILED_TILE_K) + q_off], पङ्क्तिकाः[r][kblk + slab].qs, TILED_TILE_K);
            } अन्यथा {
                स्मृतिं_पूरय(&tile->q[r * (अङ्कः_k * TILED_TILE_K) + q_off], 0, TILED_TILE_K);
            }
        }
        // d and bsums (ISA-independent)
        प्रति (पूर्णाङ्क r = 0; r < n_padded; r++) {
            यदि (r < n_पङ्क्तिकाः) {
                ध्रुव खण्डम्_q8_K & x = पङ्क्तिकाः[r][kblk + slab];
                प्रति (पूर्णाङ्क s = 0; s < n16; s++) { tile->bsums[(slab * n16 + s) * bs_stride + r] = (int32_t) x.bsums[s]; }
                tile->d[slab * TILED_MICRO + r] = x.d;
            } अन्यथा {
                प्रति (पूर्णाङ्क s = 0; s < n16; s++) { tile->bsums[(slab * n16 + s) * bs_stride + r] = 0; }
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
        ध्रुव वर्ण * env = getenv("GGML_CPU_TILED_MM");
        enabled = env == रिक्त || atoi(env) != 0;
    });
    प्रत्यागम enabled;
}

// GGML_CPU_TILED_MM_FORCE: test/bench only, take the tiled path even when unprofitable
स्थिर तार्किक ggml_tiled_matmul_forced(शून्य) {
    स्थिर तार्किक forced = असत्य;
    स्थिर std::once_flag flag;
    std::call_once(flag, []() {
        ध्रुव वर्ण * env = getenv("GGML_CPU_TILED_MM_FORCE");
        forced = env != रिक्त && atoi(env) == 1;
    });
    प्रत्यागम forced;
}

// hard constraints shared by the MUL_MAT and MUL_MAT_ID entries; the src0 type gate is the
// entries' type switch

#यदि !defined(__AVX512VNNI__) && !defined(__AVX2__) && !defined(__AVX__)
स्थिर तार्किक ggml_tiled_supported(ध्रुव संरचना ggml_tensor * src0,
                                 ध्रुव संरचना ggml_tensor * src1) {
    अप्रयुक्तम्(src0);
    अप्रयुक्तम्(src1);
    प्रत्यागम असत्य;
}
#अन्यथा
स्थिर तार्किक ggml_tiled_supported(ध्रुव संरचना ggml_tensor * src0,
                                 ध्रुव संरचना ggml_tensor * src1) {
    यदि (!ggml_tiled_matmul_enabled()) {
        प्रत्यागम असत्य;
    }

    // repack-buffer weights hold a repacked layout, let that kernel handle
    यदि (src0->extra != रिक्त) {
        प्रत्यागम असत्य;
    }
    // Supported quant types for src0
    विकल्प (src0->type) {
        पक्ष GGML_TYPE_Q6_K:
        पक्ष GGML_TYPE_Q5_K:
        पक्ष GGML_TYPE_Q4_K:
        पक्ष GGML_TYPE_Q3_K:
        पक्ष GGML_TYPE_Q2_K:
        पक्ष GGML_TYPE_IQ4_XS:
        पक्ष GGML_TYPE_IQ2_XXS:
        पक्ष GGML_TYPE_IQ2_XS:
        पक्ष GGML_TYPE_IQ2_S:
        पक्ष GGML_TYPE_IQ3_XXS:
        पक्ष GGML_TYPE_IQ3_S:
        पक्ष GGML_TYPE_IQ1_S:
        पक्ष GGML_TYPE_IQ1_M:
            प्रत्यागम सत्य;
        स्वभाव:
            प्रत्यागम असत्य;
    }
    यदि (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_Q8_K) {
        प्रत्यागम असत्य;
    }

    यदि (src1->type == GGML_TYPE_Q8_K && !ggml_is_contiguous(src1)) {
        // We can handle noncontiguous floats because we're repacking to q8_k anyways
        प्रत्यागम असत्य;
    }
    प्रत्यागम सत्य;
}
#यदि_अन्तः

// per-thread workspace slot size (0 when tiled is disabled or unsupported on this arch)
स्थिर size_t ggml_tiled_ws_आकार(शून्य) {
    यदि (!ggml_tiled_matmul_enabled()) {
        प्रत्यागम 0;
    }
    प्रत्यागम TILED_WS_SLOT; // clean 512KB slot, rounded up from sizeof(tiled_ws)
}

size_t ggml_tiled_wdata_आकार(पूर्णाङ्क n_tasks, संरचना ggml_tensor * dst) {
    यदि (! ggml_tiled_supported(dst->src[0], dst->src[1])) {
        प्रत्यागम 0; // unsupported, don't allocate
    }
    प्रत्यागम 64 + n_tasks * ggml_tiled_ws_आकार();  // 64 for alignment plus one 512KB slot per thread
}


// narrow-path K chunk: how many K's worth of weights to read contiguously into long/skinny tiles
// Assumes 32kb L1 cache budget
स्थिर पूर्णाङ्क ggml_tiled_narrow_k_extent(int64_t n_पङ्क्तिकाः, int64_t ne00) {
    पूर्णाङ्क l1 = 31 * 1024;  // fit weights in 31k to leave room for scales

    पूर्णाङ्क पङ्क्तिकाः = (पूर्णाङ्क) n_पङ्क्तिकाः;
    यदि (पङ्क्तिकाः < TILED_MICRO) पङ्क्तिकाः = TILED_MICRO;
    पूर्णाङ्क ke = l1 / (TILED_MICRO + पङ्क्तिकाः);
    ke &= ~(TILED_TILE_K - 1);  // floor to a multiple of 256
    ध्रुव पूर्णाङ्क ke_max = TILED_TILE_पङ्क्तिकाः * TILED_TILE_K / TILED_MICRO; // buffer ceiling
    यदि (ke > ke_max) ke = ke_max;
    यावत् (ke >= TILED_TILE_K && ne00 % ke != 0) {
        ke -= TILED_TILE_K;
    }
    प्रत्यागम ke < TILED_TILE_K ? 0 : ke;
}

// Writeback of the 256x256 window: buf is j-major (row stride buf_stride), dst is i-major (column stride dst_stride).
स्थिर शून्य tiled_store_गवाक्षः(ध्रुव भिन्न * buf, पूर्णाङ्क n_src0, पूर्णाङ्क n_src1, पूर्णाङ्क buf_stride,
                               भिन्न * dst, size_t dst_stride) {
    पूर्णाङ्क ri = 0;
    प्रति (; ri + 16 <= n_src0; ri += 16) {
        पूर्णाङ्क rj = 0;
        प्रति (; rj + 8 <= n_src1; rj += 8) {
            भिन्न r[16][8];
            प्रति (पूर्णाङ्क t = 0; t < 16; t++) {
                प्रति (पूर्णाङ्क u = 0; u < 8; u++) {
                    r[t][u] = buf[(ri + t) * buf_stride + rj + u];
                }
            }
            प्रति (पूर्णाङ्क u = 0; u < 8; u++) {
                प्रति (पूर्णाङ्क t = 0; t < 16; t++) {
                    dst[(ri + t) + (size_t) (rj + u) * dst_stride] = r[t][u];
                }
            }
        }
        // ragged j tail
        प्रति (; rj < n_src1; rj++) {
            प्रति (पूर्णाङ्क t = 0; t < 16; t++) {
                dst[(ri + t) + (size_t) rj * dst_stride] = buf[(ri + t) * buf_stride + rj];
            }
        }
    }
    // ragged i tail
    प्रति (; ri < n_src0; ri++) {
        प्रति (पूर्णाङ्क j = 0; j < n_src1; j++) {
            dst[ri + (size_t) j * dst_stride] = buf[ri * buf_stride + j];
        }
    }
}

// MUL_MAT_ID (MoE): src1 rows are gathered per output row (expert dispatch) and dst rows are
// scattered back. The expert's cne1 gathered q8_K rows are staged once per expert into
// thread-local scratch (contiguous rows, plus the [k/4][row][4] interleave on VNNI) so the
// unpack and microtile are reused unchanged from mul_mat.

// src0 rows per g group; finer than the TILED_TILE_K kernel tile so all threads stay busy at small ne01
#परिभाषय TILED_MMID_गणः 64

// like tiled_store_window, but the dst columns are not contiguous: col_ptrs[j] points at the
// start of dst column j, whose rows are contiguous (dim 0, nb0 == 4 bytes).
स्थिर शून्य tiled_store_गवाक्षः_scatter(ध्रुव भिन्न * buf, पूर्णाङ्क n_src0, पूर्णाङ्क n_src1, पूर्णाङ्क buf_stride,
                                       भिन्न * ध्रुव * col_ptrs) {
    पूर्णाङ्क ri = 0;
    प्रति (; ri + 16 <= n_src0; ri += 16) {
        पूर्णाङ्क rj = 0;
        प्रति (; rj + 8 <= n_src1; rj += 8) {
            भिन्न r[16][8];
            प्रति (पूर्णाङ्क t = 0; t < 16; t++) {
                प्रति (पूर्णाङ्क u = 0; u < 8; u++) {
                    r[t][u] = buf[(ri + t) * buf_stride + rj + u];
                }
            }
            प्रति (पूर्णाङ्क u = 0; u < 8; u++) {
                प्रति (पूर्णाङ्क t = 0; t < 16; t++) {
                    col_ptrs[rj + u][ri + t] = r[t][u];
                }
            }
        }
        // ragged j tail
        प्रति (; rj < n_src1; rj++) {
            प्रति (पूर्णाङ्क t = 0; t < 16; t++) {
                col_ptrs[rj][ri + t] = buf[(ri + t) * buf_stride + rj];
            }
        }
    }
    // ragged i tail
    प्रति (; ri < n_src0; ri++) {
        प्रति (पूर्णाङ्क j = 0; j < n_src1; j++) {
            col_ptrs[j][ri] = buf[ri * buf_stride + j];
        }
    }
}


// one (g, k) macrotile: zero the acc window, sweep K in 256 element slabs, scatter the result rows
// rows points at this k window's routed src1 rows, row r at its base block
प्रतिरूपम् <प्रकारनाम B, पूर्णाङ्क SUBBLK, तार्किक HAS_MIN, पूर्णाङ्क BIAS, तार्किक ACTBIAS>
स्थिर शून्य tiled_mmid_gemm_गवाक्षः(संरचना ggml_tensor * dst, ध्रुव संरचना ggml_tensor * src0,
                                   ध्रुव वर्ण * src0_cur, int64_t r, int64_t k, int64_t nrows,
                                   ध्रुव int32_t * expert_पङ्क्तिकाः,
                                   ध्रुव खण्डम्_q8_K * ध्रुव * पङ्क्तिकाः,
                                   tiled_ws * ws) {
    ध्रुव int64_t ne00 = src0->ne[0];
    ध्रुव int64_t ne01 = src0->ne[1];

    ध्रुव int64_t r_अन्तः = MIN(r + TILED_MMID_गणः, ne01);
    ध्रुव पूर्णाङ्क n_src0 = (पूर्णाङ्क) (r_अन्तः - r);

    ध्रुव size_t src0_bs = ggml_type_आकार(src0->type);
    ध्रुव int64_t src0_stride = src0->nb[1] / src0_bs;

    // the window is at most TILED_MMID_GROUP x TILED_TILE_K, so zero only that region of acc
    प्रति (int64_t i = 0; i < n_src0; i++) {
        स्मृतिं_पूरय(&ws->acc[i * TILED_TILE_पङ्क्तिकाः], 0, nrows * परिमाण(भिन्न));
    }

    // scattered writeback: column m goes to its routed dst row; r * nb[0] is the window row offset
    भिन्न * col_ptrs[TILED_TILE_पङ्क्तिकाः];
    प्रति (int64_t m = 0; m < nrows; m++) {
        col_ptrs[m] = (भिन्न *) ((वर्ण *) dst->दत्त + r * dst->nb[0] + expert_पङ्क्तिकाः[2 * (k + m) + 0] * dst->nb[1] +
                                 expert_पङ्क्तिकाः[2 * (k + m) + 1] * dst->nb[2]);
    }

    // narrow path (small nrows): we're memory bound in this case so do longer, skinny tiled in order
    // to have contiguous reads and improve memory bandwidth
    यदि (nrows <= TILED_MICRO) {
        ध्रुव पूर्णाङ्क k_extent = ggml_tiled_narrow_k_extent(nrows, ne00);
        ध्रुव पूर्णाङ्क अङ्कः_k = k_extent / TILED_TILE_K;
        प्रति (int64_t k0 = 0; k0 < ne00; k0 += k_extent) {
            ध्रुव पूर्णाङ्क kstart = (पूर्णाङ्क) (k0 / TILED_TILE_K);
            // activation: one 16-row band (nrows <= 16). Unpack the whole k_extent chunk (all
            // slabs) so each activation row is a long stream, then repack each slab in place
            tiled_unpack_src1_q8_K(पङ्क्तिकाः, nrows, &ws->src1, kstart, अङ्कः_k);
            // repack the whole k_extent chunk in place: num_k slabs x 4 tiles, 16 rows, row stride num_k*256
            tiled_repack_src1(&ws->src1, 0, अङ्कः_k, ACTBIAS);
            // weight groups (16 at a time): unpack the k_extent chunk of the group (the long per-row
            // read), then one standard MAC per slab accumulating into the same acc rows
            प्रति (int64_t ir0 = r; ir0 < r_अन्तः; ir0 += TILED_MICRO) {
                ध्रुव पूर्णाङ्क n0 = (पूर्णाङ्क) MIN(TILED_MICRO, r_अन्तः - ir0);
                ध्रुव B * wbase = (ध्रुव B *) (src0_cur + ir0 * src0->nb[1] + kstart * src0_bs);
                tiled_unpack_src0(wbase, src0_stride, n0, &ws->src0, अङ्कः_k);
                tiled_repack_src0<SUBBLK>(&ws->src0, n0, अङ्कः_k, BIAS, ACTBIAS);
                भिन्न * buf = ws->acc + (ir0 - r) * TILED_TILE_पङ्क्तिकाः;
                प्रति (पूर्णाङ्क slab = 0; slab < अङ्कः_k; slab++) {
                    tiled_सततम्_microtile<SUBBLK, HAS_MIN, BIAS, ACTBIAS>(ws->src0, ws->src1, 0, 0, अङ्कः_k, slab, buf, TILED_TILE_पङ्क्तिकाः);
                }
            }
        }
    } अन्यथा {
        // standard per-slab path: K is stepped in 256-element slabs
        प्रति (int64_t ib = 0; ib < ne00; ib += TILED_TILE_K) {
            ध्रुव पूर्णाङ्क kblk = (पूर्णाङ्क) (ib / TILED_TILE_K);
            tiled_unpack_src0((ध्रुव B *) (src0_cur + r * src0->nb[1] + kblk * src0_bs), src0_stride, n_src0, &ws->src0, 1);
            tiled_repack_src0<SUBBLK>(&ws->src0, n_src0, 1, BIAS, ACTBIAS);
            tiled_unpack_src1_q8_K(पङ्क्तिकाः, nrows, &ws->src1, kblk, 1);
            // 16x16 microtiles sweeping the window; repack each src1 band just-in-time
            // (j0-outer) so only the bands actually used are repacked and each is L1-hot
            प्रति (int64_t ir1 = 0; ir1 < nrows; ir1 += TILED_MICRO) {
                tiled_repack_src1(&ws->src1, (पूर्णाङ्क) ir1, 1, ACTBIAS);
                प्रति (int64_t ir0 = r; ir0 < r_अन्तः; ir0 += TILED_MICRO) {
                    tiled_सततम्_microtile<SUBBLK, HAS_MIN, BIAS, ACTBIAS>(ws->src0, ws->src1,
                        (पूर्णाङ्क) (ir0 - r), (पूर्णाङ्क) ir1, 1, 0,
                        ws->acc, TILED_TILE_पङ्क्तिकाः);
                }
            }
        }
    }

    tiled_store_गवाक्षः_scatter(ws->acc, n_src0, nrows, TILED_TILE_पङ्क्तिकाः, col_ptrs);
}

// one expert of MUL_MAT_ID: each k window's dispatched rows are pointed at by a per-window
// row pointer list and swept over the thread's row windows, the dst rows are scattered back
// per window
प्रतिरूपम् <प्रकारनाम B, पूर्णाङ्क SUBBLK, तार्किक HAS_MIN, पूर्णाङ्क BIAS, तार्किक ACTBIAS>
स्थिर शून्य ggml_compute_forward_mul_mat_id_tiled_one_expert(
        ध्रुव संरचना ggml_compute_params * params,
              संरचना ggml_tensor *         dst,
        int64_t                            cur_a,
        int64_t                            cne1,
        ध्रुव int32_t *                    expert_पङ्क्तिकाः,
        वर्ण *                             scratch) {

    ध्रुव संरचना ggml_tensor * src0 = dst->src[0];
    ध्रुव संरचना ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव गणना ggml_type vec_dot_type = ggml_get_type_traits_cpu(src0->type)->vec_dot_type;

    ध्रुव size_t nbw1 = (src1->type == vec_dot_type) ? nb11 : ggml_row_आकार(vec_dot_type, ne10);
    ध्रुव शून्य * wdata = (src1->type == vec_dot_type) ? src1->दत्त : params->wdata;

    ध्रुव वर्ण * src0_cur = (ध्रुव वर्ण *) src0->दत्त + cur_a * nb02;

    // this thread's workspace
    tiled_ws * ws = (tiled_ws *) (scratch + (size_t) ith * ggml_tiled_ws_आकार());

    // groups of TILED_MMID_GROUP rows; rounded up, the window tail is clamped in the gemm
    ध्रुव int64_t ngroups = (ne01 + TILED_MMID_गणः - 1) / TILED_MMID_गणः;

    ध्रुव int64_t g0 = (ngroups * ith) / nth;
    ध्रुव int64_t g1 = (ngroups * (ith + 1)) / nth;

    // no rows for this thread; nothing to do
    यदि (g0 >= g1) {
        प्रत्यागम;
    }

    // MUL_MAT_ID scatters the src1 rows per routed row: point each k window's rows
    // at the routed src1 rows, then sweep the row windows
    प्रति (int64_t k = 0; k < cne1; k += TILED_TILE_K) {
        ध्रुव int64_t nrows = MIN(TILED_TILE_K, cne1 - k);

        ध्रुव खण्डम्_q8_K * पङ्क्तिकाः[TILED_TILE_पङ्क्तिकाः];
        प्रति (int64_t i = 0; i < nrows; i++) {
            ध्रुव int64_t i11 = expert_पङ्क्तिकाः[2 * (k + i) + 0] % ne11;
            ध्रुव int64_t i12 = expert_पङ्क्तिकाः[2 * (k + i) + 1];
            पङ्क्तिकाः[i] = (ध्रुव खण्डम्_q8_K *) ((ध्रुव वर्ण *) wdata + (i11 + i12 * ne11) * nbw1);
        }

        प्रति (int64_t g = g0; g < g1; g++) {
            ध्रुव int64_t r = g * TILED_MMID_गणः;

            tiled_mmid_gemm_गवाक्षः<B, SUBBLK, HAS_MIN, BIAS, ACTBIAS>(dst, src0, src0_cur, r, k, nrows, expert_पङ्क्तिकाः, पङ्क्तिकाः, ws);
        }
    }
}

प्रतिरूपम् <प्रकारनाम B, पूर्णाङ्क SUBBLK, तार्किक HAS_MIN, पूर्णाङ्क BIAS, तार्किक ACTBIAS>
स्थिर शून्य ggml_compute_forward_mul_mat_tiled_one_chunk(
    ध्रुव संरचना ggml_compute_params * params,
    संरचना ggml_tensor * dst,
    ध्रुव int64_t ir0_start,
    ध्रुव int64_t ir0_अन्तः,
    ध्रुव int64_t ir1_start,
    ध्रुव int64_t ir1_अन्तः,
    tiled_ws * ws) {

    ध्रुव संरचना ggml_tensor * src0 = dst->src[0];
    ध्रुव संरचना ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS

    ध्रुव गणना ggml_type vec_dot_type = ggml_get_type_traits_cpu(src0->type)->vec_dot_type;

    // broadcast factors
    ध्रुव int64_t r2 = ne12 / ne02;
    ध्रुव int64_t r3 = ne13 / ne03;

    यदि (ir0_start >= ir0_अन्तः || ir1_start >= ir1_अन्तः) {
        प्रत्यागम;
    }

    ध्रुव शून्य * wdata = (src1->type == vec_dot_type) ? src1->दत्त : params->wdata;
    ध्रुव size_t row_आकार = ggml_row_आकार(vec_dot_type, ne10);
    ध्रुव size_t src0_bs  = ggml_type_आकार(src0->type);
    ध्रुव size_t src1_bs  = ggml_type_आकार(vec_dot_type);

    GGML_ASSERT(ne00 % 256 == 0);
    प्रतिज्ञा(ne12 % ne02 == 0);
    प्रतिज्ञा(ne13 % ne03 == 0);

    ध्रुव int64_t src0_stride = nb01 / src0_bs;  // blocks between src0 rows
    ध्रुव int64_t src1_stride = (src1->type == vec_dot_type ? src1->nb[1] : row_आकार) / src1_bs;

    ध्रुव int64_t TILE = 256;
    ध्रुव int64_t MICRO = 16;

    // 256-wide windows over the chunk. The iir1 window is additionally clamped at the
    // src1 batch (ne11) boundary: the tiles require a constant batch index (i12/i13)
    // within a window, so advance by the clamped end, not a fixed 256.
    प्रति (int64_t iir1 = ir1_start; iir1 < ir1_अन्तः; ) {
        int64_t iir1_अन्तः = MIN(iir1 + TILE, ir1_अन्तः);
        ध्रुव int64_t bnd = (iir1 / ne11 + 1) * ne11;
        यदि (bnd < iir1_अन्तः) {
            iir1_अन्तः = bnd;
        }

        ध्रुव पूर्णाङ्क n_src1 = (पूर्णाङ्क) (iir1_अन्तः - iir1);

        // batch coords, constant within the clamped window
        ध्रुव int64_t i13 = iir1 / (ne12 * ne11);
        ध्रुव int64_t i12 = (iir1 - i13 * ne12 * ne11) / ne11;
        // within-batch row; dst_col below holds the i12/i13 batch offset, so the store
        // applies i11 * nb1 (not the flattened iir1, which spans all batch dims)
        ध्रुव int64_t i11 = iir1 - i13 * ne12 * ne11 - i12 * ne11;

        // dst batches == src1 batches (ggml_mul_mat), so the loop batch coords are in
        // src1 space; src0 batches are broadcast over them, map down into src0's batch
        ध्रुव int64_t i02 = i12 / r2;
        ध्रुव int64_t i03 = i13 / r3;

        ध्रुव वर्ण * src0_row = (ध्रुव वर्ण *) src0->दत्त + i02 * src0->nb[2] + i03 * src0->nb[3];
        वर्ण * dst_col = (वर्ण *) dst->दत्त + i12 * nb2 + i13 * nb3;

        // rows[r] is the window's row r base block; the k-slab offset is applied in the unpack
        ध्रुव खण्डम्_q8_K * पङ्क्तिकाः[TILED_TILE_पङ्क्तिकाः];
        प्रति (पूर्णाङ्क r = 0; r < n_src1; r++) {
            पङ्क्तिकाः[r] = (ध्रुव खण्डम्_q8_K *) ((ध्रुव वर्ण *) wdata + (iir1 + r) * src1_stride * src1_bs);
        }

        प्रति (int64_t iir0 = ir0_start; iir0 < ir0_अन्तः; iir0 += TILE) {
            int64_t iir0_अन्तः = MIN(iir0 + TILE, ir0_अन्तः);
            ध्रुव पूर्णाङ्क n_src0 = (पूर्णाङ्क) (iir0_अन्तः - iir0);

            // result buffer zeroed once per macrotile
            स्मृतिं_पूरय(ws->acc, 0, (size_t)TILED_TILE_पङ्क्तिकाः * TILED_TILE_पङ्क्तिकाः * परिमाण(भिन्न));

            यदि (n_src1 <= TILED_MICRO) {
                // narrow path: if we have few enough rows, we're memory bound
                // Do longer, skinner tiles so we have longer contiguous reads and improve bandwidth
                ध्रुव पूर्णाङ्क k_extent = ggml_tiled_narrow_k_extent(n_src1, ne00);
                ध्रुव पूर्णाङ्क अङ्कः_k = k_extent / TILED_TILE_K;
                प्रति (int64_t k0 = 0; k0 < ne00; k0 += k_extent) {
                    ध्रुव पूर्णाङ्क kstart = (पूर्णाङ्क) (k0 / TILED_TILE_K);
                    // activation: one 16-row band (n_src1 <= 16). Unpack the whole k_extent chunk
                    // (all slabs) so each activation row is a long stream, then repack each slab in place
                    tiled_unpack_src1_q8_K(पङ्क्तिकाः, n_src1, &ws->src1, kstart, अङ्कः_k);
                    // repack the whole k_extent chunk in place, if kernel wants to: num_k slabs x 4 tiles, 16 rows, row stride num_k*256
                    tiled_repack_src1(&ws->src1, 0, अङ्कः_k, ACTBIAS);
                    // weight groups: unpack the k_extent chunk of each 16-row group (the long per-row
                    // read), then one standard MAC per slab accumulating into the same acc rows
                    प्रति (int64_t ir0 = iir0; ir0 < iir0_अन्तः; ir0 += MICRO) {
                        ध्रुव पूर्णाङ्क n0 = (पूर्णाङ्क) MIN(MICRO, iir0_अन्तः - ir0);
                        ध्रुव B * wbase = (ध्रुव B *) (src0_row + ir0 * nb01 + kstart * src0_bs);
                        tiled_unpack_src0(wbase, src0_stride, n0, &ws->src0, अङ्कः_k);
                        tiled_repack_src0<SUBBLK>(&ws->src0, n0, अङ्कः_k, BIAS, ACTBIAS);
                        भिन्न * buf = ws->acc + (ir0 - iir0) * TILED_TILE_पङ्क्तिकाः;
                        प्रति (पूर्णाङ्क slab = 0; slab < अङ्कः_k; slab++) {
                            tiled_सततम्_microtile<SUBBLK, HAS_MIN, BIAS, ACTBIAS>(ws->src0, ws->src1, 0, 0, अङ्कः_k, slab, buf, TILED_TILE_पङ्क्तिकाः);
                        }
                    }
                }
            } अन्यथा {
                // standard per-slab path: K is stepped in 256-element chunks
                प्रति (int64_t ib = 0; ib < ne00; ib += TILE) {
                    ध्रुव पूर्णाङ्क kblk = (पूर्णाङ्क) (ib / TILE);
                    tiled_unpack_src0((ध्रुव B *) (src0_row + iir0 * nb01 + kblk * src0_bs), src0_stride, n_src0, &ws->src0, 1);
                    tiled_repack_src0<SUBBLK>(&ws->src0, n_src0, 1, BIAS, ACTBIAS);
                    tiled_unpack_src1_q8_K(पङ्क्तिकाः, n_src1, &ws->src1, kblk, 1);

                    // 16x16 microtiles sweeping the window; repack each src1 band before first use
                    प्रति (int64_t ir1 = iir1; ir1 < iir1_अन्तः; ir1 += MICRO) {
                        tiled_repack_src1(&ws->src1, (पूर्णाङ्क) (ir1 - iir1), 1, ACTBIAS);
                        प्रति (int64_t ir0 = iir0; ir0 < iir0_अन्तः; ir0 += MICRO) {
                            tiled_सततम्_microtile<SUBBLK, HAS_MIN, BIAS, ACTBIAS>(ws->src0, ws->src1,
                                (पूर्णाङ्क) (ir0 - iir0), (पूर्णाङ्क) (ir1 - iir1), 1, 0,
                                ws->acc, TILED_TILE_पङ्क्तिकाः);
                        }
                    }
                }
            }
            // write acc back out from L2 to main memory
            tiled_store_गवाक्षः(ws->acc, n_src0, n_src1, TILED_TILE_पङ्क्तिकाः,
                               (भिन्न *) (dst_col + iir0 * nb0 + i11 * nb1), nb1 / nb0);
        }
        iir1 = iir1_अन्तः;
    }
}

प्रतिरूपम् <प्रकारनाम B, पूर्णाङ्क SUBBLK, तार्किक HAS_MIN, पूर्णाङ्क BIAS, तार्किक ACTBIAS>
स्थिर शून्य ggml_compute_forward_mul_mat_tiled_driver(
        ध्रुव संरचना ggml_compute_params * params,
              संरचना ggml_tensor * dst) {

    ध्रुव संरचना ggml_tensor * src0 = dst->src[0];
    ध्रुव संरचना ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    गणना ggml_type      ध्रुव vec_dot_type = ggml_get_type_traits_cpu(src0->type)->vec_dot_type;
    ggml_from_float_t   ध्रुव from_float   = ggml_get_type_traits_cpu(vec_dot_type)->from_float;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb10 == ggml_type_आकार(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == परिमाण(भिन्न));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    यदि (src1->type != vec_dot_type) {
        वर्ण * wdata = (वर्ण *) params->wdata;

        ध्रुव size_t nbw0 = ggml_type_आकार(vec_dot_type);
        ध्रुव size_t nbw1 = ggml_row_आकार(vec_dot_type, ne10);
        ध्रुव size_t nbw2 = nbw1*ne11;
        ध्रुव size_t nbw3 = nbw2*ne12;

        प्रतिज्ञा(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        प्रति (int64_t i13 = 0; i13 < ne13; ++i13) {
            प्रति (int64_t i12 = 0; i12 < ne12; ++i12) {
                प्रति (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_आकार(vec_dot_type);
                    int64_t ne10_खण्डम्_start = (ith * ne10/bs) / nth;
                    int64_t ne10_खण्डम्_अन्तः   = ((ith + 1) * ne10/bs) / nth;
                    from_float((भिन्न *)((वर्ण *) src1->दत्त + i13*src1->nb[3] + i12*src1->nb[2] + i11*src1->nb[1] + ne10_खण्डम्_start*bs*src1->nb[0]),
                               (शून्य *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_खण्डम्_start*nbw0),
                               (ne10_खण्डम्_अन्तः - ne10_खण्डम्_start) * bs);
                }
            }
        }
    }

    यदि (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth. This saves a bit of coordination right at the start.
        ggml_threadpool_chunk_समुच्चयः(params->threadpool, nth);
    }

    ggml_barrier(params->threadpool);

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    ध्रुव int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    ध्रुव int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    पूर्णाङ्क chunk_आकार = 256; //TILED_TILE_ROWS;

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim. CEIL(nr/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_आकार - 1) / chunk_आकार;
    int64_t nchunk1 = (nr1 + chunk_आकार - 1) / chunk_आकार;

    // Step down chunk size if too few chunks to saturate cores, minimum is microtile size
    यावत् (nchunk0 * nchunk1 < nth * 4 && chunk_आकार > 16) {
        chunk_आकार = chunk_आकार / 2;
        nchunk0 = (nr0 + chunk_आकार - 1) / chunk_आकार;
        nchunk1 = (nr1 + chunk_आकार - 1) / chunk_आकार;
    }

    // The number of elements in each chunk
    ध्रुव int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    ध्रुव int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    पूर्णाङ्क current_chunk = ith;

    // per-thread workspace: after any converted src1 data in wdata, aligned to 64
    वर्ण * ws_आधारः = (वर्ण *) params->wdata;
    यदि (src1->type != vec_dot_type) {
        ws_आधारः += GGML_PAD(ggml_row_आकार(vec_dot_type, ggml_nelements(src1)), 64);
    }
    ws_आधारः = (वर्ण *) (((uintptr_t) ws_आधारः + 63) & ~(uintptr_t) 63);
    tiled_ws * ws = (tiled_ws *) (ws_आधारः + (size_t) ith * ggml_tiled_ws_आकार());

    // TODO:  if we KNOW we're on a machine where all cores are equal, we could skip the coordination/work-stealing and just assign chunks deterministically
    यावत् (current_chunk < nchunk0 * nchunk1) {
        ध्रुव int64_t ith0 = current_chunk % nchunk0;
        ध्रुव int64_t ith1 = current_chunk / nchunk0;

        ध्रुव int64_t ir0_start = dr0 * ith0;
        ध्रुव int64_t ir0_अन्तः = MIN(ir0_start + dr0, nr0);

        ध्रुव int64_t ir1_start = dr1 * ith1;
        ध्रुव int64_t ir1_अन्तः = MIN(ir1_start + dr1, nr1);

        ggml_compute_forward_mul_mat_tiled_one_chunk<B, SUBBLK, HAS_MIN, BIAS, ACTBIAS>(params, dst, ir0_start, ir0_अन्तः, ir1_start, ir1_अन्तः, ws);

        यदि (nth >= nchunk0 * nchunk1) {
            भङ्ग;
        }

        current_chunk = ggml_threadpool_chunk_सङ्कलय(params->threadpool, 1);
    }
}

// src0 type dispatch, shared by the MUL_MAT and MUL_MAT_ID entries: one expert for
// MUL_MAT_ID (expert_rows != NULL), the full op for MUL_MAT
प्रतिरूपम् <प्रकारनाम B, पूर्णाङ्क SUBBLK, तार्किक HAS_MIN, पूर्णाङ्क BIAS, तार्किक ACTBIAS>
स्थिर तार्किक tiled_matmul_dispatch(ध्रुव संरचना ggml_compute_params * params,
                                  संरचना ggml_tensor * dst,
                                  ध्रुव int32_t * expert_पङ्क्तिकाः,
                                  int64_t cur_a,
                                  int64_t cne1,
                                  वर्ण * scratch) {
    यदि (expert_पङ्क्तिकाः == रिक्त) {
        ggml_compute_forward_mul_mat_tiled_driver<B, SUBBLK, HAS_MIN, BIAS, ACTBIAS>(params, dst);
    } अन्यथा {
        ggml_compute_forward_mul_mat_id_tiled_one_expert<B, SUBBLK, HAS_MIN, BIAS, ACTBIAS>(params, dst, cur_a, cne1, expert_पङ्क्तिकाः, scratch);
    }
    प्रत्यागम सत्य;
}

// the supported src0 types, one list for both ops; false to fall through to vec_dot
स्थिर तार्किक ggml_tiled_matmul_type_dispatch(ध्रुव संरचना ggml_compute_params * params,
                                            संरचना ggml_tensor * dst,
                                            ध्रुव int32_t * expert_पङ्क्तिकाः = रिक्त,
                                            int64_t cur_a = 0,
                                            int64_t cne1 = 0,
                                            वर्ण * scratch = रिक्त) {
    विकल्प (dst->src[0]->type) {
        पक्ष GGML_TYPE_Q6_K:
            प्रत्यागम tiled_matmul_dispatch<खण्डम्_q6_K, 16, असत्य, 32, सत्य>(params, dst, expert_पङ्क्तिकाः, cur_a, cne1, scratch);
        पक्ष GGML_TYPE_Q5_K:
            प्रत्यागम tiled_matmul_dispatch<खण्डम्_q5_K, 32, सत्य,  0, असत्य>(params, dst, expert_पङ्क्तिकाः, cur_a, cne1, scratch);
        पक्ष GGML_TYPE_Q4_K:
            प्रत्यागम tiled_matmul_dispatch<खण्डम्_q4_K, 32, सत्य,  0, असत्य>(params, dst, expert_पङ्क्तिकाः, cur_a, cne1, scratch);
        पक्ष GGML_TYPE_Q3_K:
            प्रत्यागम tiled_matmul_dispatch<खण्डम्_q3_K, 16, असत्य,  4, सत्य>(params, dst, expert_पङ्क्तिकाः, cur_a, cne1, scratch);
        पक्ष GGML_TYPE_Q2_K:
            प्रत्यागम tiled_matmul_dispatch<खण्डम्_q2_K, 16, सत्य,  0, असत्य>(params, dst, expert_पङ्क्तिकाः, cur_a, cne1, scratch);
        पक्ष GGML_TYPE_IQ4_XS:
            प्रत्यागम tiled_matmul_dispatch<खण्डम्_iq4_xs, 32, असत्य, 128, असत्य>(params, dst, expert_पङ्क्तिकाः, cur_a, cne1, scratch);
        पक्ष GGML_TYPE_IQ2_XXS:
            प्रत्यागम tiled_matmul_dispatch<खण्डम्_iq2_xxs, 32, असत्य, 128, सत्य>(params, dst, expert_पङ्क्तिकाः, cur_a, cne1, scratch);
        पक्ष GGML_TYPE_IQ2_XS:
            प्रत्यागम tiled_matmul_dispatch<खण्डम्_iq2_xs, 16, असत्य, 128, सत्य>(params, dst, expert_पङ्क्तिकाः, cur_a, cne1, scratch);
        पक्ष GGML_TYPE_IQ2_S:
            प्रत्यागम tiled_matmul_dispatch<खण्डम्_iq2_s, 16, असत्य, 128, सत्य>(params, dst, expert_पङ्क्तिकाः, cur_a, cne1, scratch);
        पक्ष GGML_TYPE_IQ3_XXS:
            प्रत्यागम tiled_matmul_dispatch<खण्डम्_iq3_xxs, 32, असत्य, 128, सत्य>(params, dst, expert_पङ्क्तिकाः, cur_a, cne1, scratch);
        पक्ष GGML_TYPE_IQ3_S:
            प्रत्यागम tiled_matmul_dispatch<खण्डम्_iq3_s, 32, असत्य, 128, सत्य>(params, dst, expert_पङ्क्तिकाः, cur_a, cne1, scratch);
        पक्ष GGML_TYPE_IQ1_S:
            प्रत्यागम tiled_matmul_dispatch<खण्डम्_iq1_s, 32, असत्य, 128, सत्य>(params, dst, expert_पङ्क्तिकाः, cur_a, cne1, scratch);
        पक्ष GGML_TYPE_IQ1_M:
            प्रत्यागम tiled_matmul_dispatch<खण्डम्_iq1_m, 16, असत्य, 128, सत्य>(params, dst, expert_पङ्क्तिकाः, cur_a, cne1, scratch);
        स्वभाव:
            प्रत्यागम असत्य;
    }
}

स्थिर तार्किक ggml_tiled_min_batch(int64_t पङ्क्तिकाः) {
    //  Profitable at rows >= 8, take even when unprofitable if we're forced
    प्रत्यागम पङ्क्तिकाः >= 8 || ggml_tiled_matmul_forced();
}

// tiled K-quant matmul; returns true if the op was computed here,
// false to fall through to the stock path
तार्किक ggml_compute_forward_mul_mat_tiled(
        ध्रुव संरचना ggml_compute_params * params,
              संरचना ggml_tensor * dst) {
    // --use-ref means bail out and go back to vec_dot reference impl
    यदि (params->use_ref) {
        प्रत्यागम असत्य;
    }
    यदि (!ggml_tiled_supported(dst->src[0], dst->src[1])) {
        प्रत्यागम असत्य;
    }
    यदि (!ggml_tiled_min_batch(dst->src[1]->ne[1])) {
        प्रत्यागम असत्य;
    }
    प्रत्यागम ggml_tiled_matmul_type_dispatch(params, dst);
}

// MUL_MAT_ID (MoE), one expert; returns true if the expert was computed here,
// per expert eligibility (type gate, batch floor) is decided here
तार्किक ggml_compute_forward_mul_mat_id_tiled(
        ध्रुव संरचना ggml_compute_params * params,
              संरचना ggml_tensor *         dst,
        int64_t                            cur_a,
        int64_t                            cne1,
        ध्रुव int32_t *                    expert_पङ्क्तिकाः,
        वर्ण *                             scratch) {
    यदि (params->use_ref) {
        प्रत्यागम असत्य;
    }
    यदि (!ggml_tiled_supported(dst->src[0], dst->src[1])) {
        प्रत्यागम असत्य;
    }
    // profitability is per expert: the rows routed to this expert
    यदि (!ggml_tiled_min_batch(cne1)) {
        प्रत्यागम असत्य;
    }
    प्रत्यागम ggml_tiled_matmul_type_dispatch(params, dst, expert_पङ्क्तिकाः, cur_a, cne1, scratch);
}
