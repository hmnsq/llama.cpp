#అన్తర్భావ "tiled.h"
#అన్తర్భావ "tiled-kernel.h"

#అన్తర్భావ "ggml-cpu-impl.h"
#అన్తర్భావ "ggml-cpu.h"
#అన్తర్భావ "ggml.h"

// kvalues table (impl section) for the iq4_xs unpack
#నిర్వచించు GGML_COMMON_IMPL_CPP
#అన్తర్భావ "ggml-common.h"

#అన్తర్భావ <cassert>
#అన్తర్భావ <cstdlib>
#అన్తర్భావ <cstring>

#అన్తర్భావ <mutex>

#నిర్వచించు వాడని GGML_వాడని

// unpack routines for various quant types src0
స్థిర శూన్య tiled_unpack_src0(స్థిరరాశి ఖండం_q4_K * అడ్డువరుసలు, int64_t అడ్డువరుస_stride, పూర్ణాంకం n_అడ్డువరుసలు, tiled_tile_src0 * tile, పూర్ణాంకం అంకె_k) {
    GGML_ASSERT(n_అడ్డువరుసలు <= TILED_TILE_అడ్డువరుసలు);
    constexpr పూర్ణాంకం NB = 8; // 32-wide subblocks
    // 12-byte packed scale/min decode, same extraction as the reference kernels
    స్థిర స్థిరరాశి uint32_t kmask1 = 0x3f3f3f3f;
    స్థిర స్థిరరాశి uint32_t kmask2 = 0x0f0f0f0f;
    స్థిర స్థిరరాశి uint32_t kmask3 = 0x03030303;

    స్థిరరాశి పూర్ణాంకం qk_stride = అంకె_k * TILED_TILE_K;
    స్థిరరాశి పూర్ణాంకం nb_stride = NB * అంకె_k;
    కొరకు (పూర్ణాంకం slab = 0; slab < అంకె_k; slab++) {
        కొరకు (పూర్ణాంకం r = 0; r < n_అడ్డువరుసలు; r++) {
            స్థిరరాశి పూర్ణాంకం d_off = slab * TILED_MICRO + r;
            స్థిరరాశి పూర్ణాంకం q_off = r * qk_stride + slab * TILED_TILE_K;
            స్థిరరాశి పూర్ణాంకం s_off = r * nb_stride + slab * NB;
            స్థిరరాశి ఖండం_q4_K & x = అడ్డువరుసలు[r * అడ్డువరుస_stride + slab];

            tile->d[d_off]    = ggml_fp16_గమ్యం_fp32(x.దత్తాంశం.దత్తాంశం.d);
            tile->dmin[d_off] = ggml_fp16_గమ్యం_fp32(x.దత్తాంశం.దత్తాంశం.dmin);

            uint32_t utmp[4];
            జ్ఞాపకం_నకలు(utmp, x.scales, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            స్థిరరాశి uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;

            స్థిరరాశి uint8_t * scales = (స్థిరరాశి uint8_t *) &utmp[0];
            స్థిరరాశి uint8_t * mins = (స్థిరరాశి uint8_t *) &utmp[2];
            కొరకు (పూర్ణాంకం s = 0; s < NB; s++) {
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

స్థిర శూన్య tiled_unpack_src0(స్థిరరాశి ఖండం_q5_K * అడ్డువరుసలు, int64_t అడ్డువరుస_stride, పూర్ణాంకం n_అడ్డువరుసలు, tiled_tile_src0 * tile, పూర్ణాంకం అంకె_k) {
    GGML_ASSERT(n_అడ్డువరుసలు <= TILED_TILE_అడ్డువరుసలు);
    constexpr పూర్ణాంకం NB = 8; // 32-wide subblocks
    // 12-byte packed scale/min decode, same extraction as the reference kernels
    స్థిర స్థిరరాశి uint32_t kmask1 = 0x3f3f3f3f;
    స్థిర స్థిరరాశి uint32_t kmask2 = 0x0f0f0f0f;
    స్థిర స్థిరరాశి uint32_t kmask3 = 0x03030303;

    స్థిరరాశి పూర్ణాంకం qk_stride = అంకె_k * TILED_TILE_K;
    స్థిరరాశి పూర్ణాంకం nb_stride = NB * అంకె_k;
    కొరకు (పూర్ణాంకం slab = 0; slab < అంకె_k; slab++) {
        కొరకు (పూర్ణాంకం r = 0; r < n_అడ్డువరుసలు; r++) {
            స్థిరరాశి పూర్ణాంకం d_off = slab * TILED_MICRO + r;
            స్థిరరాశి పూర్ణాంకం q_off = r * qk_stride + slab * TILED_TILE_K;
            స్థిరరాశి పూర్ణాంకం s_off = r * nb_stride + slab * NB;
            స్థిరరాశి ఖండం_q5_K & x = అడ్డువరుసలు[r * అడ్డువరుస_stride + slab];

            tile->d[d_off]    = ggml_fp16_గమ్యం_fp32(x.దత్తాంశం.దత్తాంశం.d);
            tile->dmin[d_off] = ggml_fp16_గమ్యం_fp32(x.దత్తాంశం.దత్తాంశం.dmin);

            uint32_t utmp[4];
            జ్ఞాపకం_నకలు(utmp, x.scales, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            స్థిరరాశి uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;

            స్థిరరాశి uint8_t * scales = (స్థిరరాశి uint8_t *) &utmp[0];
            స్థిరరాశి uint8_t * mins = (స్థిరరాశి uint8_t *) &utmp[2];
            కొరకు (పూర్ణాంకం s = 0; s < NB; s++) {
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

స్థిర శూన్య tiled_unpack_src0(స్థిరరాశి ఖండం_q6_K * అడ్డువరుసలు, int64_t అడ్డువరుస_stride, పూర్ణాంకం n_అడ్డువరుసలు, tiled_tile_src0 * tile, పూర్ణాంకం అంకె_k) {
    GGML_ASSERT(n_అడ్డువరుసలు <= TILED_TILE_అడ్డువరుసలు);
    constexpr పూర్ణాంకం NB = 16; // 16-wide subblocks
    స్థిరరాశి పూర్ణాంకం qk_stride = అంకె_k * TILED_TILE_K;
    స్థిరరాశి పూర్ణాంకం nb_stride = NB * అంకె_k;
    కొరకు (పూర్ణాంకం slab = 0; slab < అంకె_k; slab++) {
        కొరకు (పూర్ణాంకం r = 0; r < n_అడ్డువరుసలు; r++) {
            స్థిరరాశి పూర్ణాంకం d_off = slab * TILED_MICRO + r;
            స్థిరరాశి పూర్ణాంకం q_off = r * qk_stride + slab * TILED_TILE_K;
            స్థిరరాశి పూర్ణాంకం s_off = r * nb_stride + slab * NB;
            స్థిరరాశి ఖండం_q6_K & x = అడ్డువరుసలు[r * అడ్డువరుస_stride + slab];
            tile->d[d_off] = ggml_fp16_గమ్యం_fp32(x.d);

            // 6-bit code = 4 low bits (ql) | 2 high bits (qh); see ggml_vec_dot_q6_K_q8_K_generic
            // per half the lanes are [ql lo(0:32)] [ql lo(32:64)] [ql hi(0:32)] [ql hi(32:64)]
            uint8_t * q = &tile->q[q_off];
            కొరకు (పూర్ణాంకం half = 0; half < 2; half++) {
                uint8_t * బహిర్గతం = q + 128 * half;
                tiled_unpk_nib4(x.ql + 64 * half + 0,  బహిర్గతం + 0,  బహిర్గతం + 64);
                tiled_unpk_nib4(x.ql + 64 * half + 32, బహిర్గతం + 32, బహిర్గతం + 96);
                tiled_unpk_or<0, 4, 3>(బహిర్గతం + 0,  x.qh + 32 * half);
                tiled_unpk_or<2, 4, 3>(బహిర్గతం + 32, x.qh + 32 * half);
                tiled_unpk_or<4, 4, 3>(బహిర్గతం + 64, x.qh + 32 * half);
                tiled_unpk_or<6, 4, 3>(బహిర్గతం + 96, x.qh + 32 * half);
            }
            // scale is a plain int8 per 16-element subblock (16 per 256-K)
            కొరకు (పూర్ణాంకం s = 0; s < NB; s++) { tile->scales[s_off + s] = (int32_t) (int8_t) x.scales[s]; }
        }
    }
}

స్థిర శూన్య tiled_unpack_src0(స్థిరరాశి ఖండం_q3_K * అడ్డువరుసలు, int64_t అడ్డువరుస_stride, పూర్ణాంకం n_అడ్డువరుసలు, tiled_tile_src0 * tile, పూర్ణాంకం అంకె_k) {
    GGML_ASSERT(n_అడ్డువరుసలు <= TILED_TILE_అడ్డువరుసలు);
    constexpr పూర్ణాంకం NB = 16; // 16-wide subblocks
    స్థిరరాశి పూర్ణాంకం qk_stride = అంకె_k * TILED_TILE_K;
    స్థిరరాశి పూర్ణాంకం nb_stride = NB * అంకె_k;
    కొరకు (పూర్ణాంకం slab = 0; slab < అంకె_k; slab++) {
        కొరకు (పూర్ణాంకం r = 0; r < n_అడ్డువరుసలు; r++) {
            స్థిరరాశి పూర్ణాంకం d_off = slab * TILED_MICRO + r;
            స్థిరరాశి పూర్ణాంకం q_off = r * qk_stride + slab * TILED_TILE_K;
            స్థిరరాశి పూర్ణాంకం s_off = r * nb_stride + slab * NB;
            స్థిరరాశి ఖండం_q3_K & x = అడ్డువరుసలు[r * అడ్డువరుస_stride + slab];
            tile->d[d_off] = ggml_fp16_గమ్యం_fp32(x.d);

            // 3-bit code = 2 low bits (qs) | (1 high bit from hmask << 2)
            // element e (0..255): half=e>>7, el=e&127, group=el>>5, l=el&31
            //   low2 = (qs[half*32 + l] >> 2*group) & 3
            //   high = (hmask[l] >> (half*4 + group)) & 1
            // see ggml_vec_dot_q3_K_q8_K_generic
            uint8_t * q = &tile->q[q_off];
            స్థిరరాశి uint8_t * s0 = x.qs;
            స్థిరరాశి uint8_t * s1 = x.qs + 32;
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
            స్థిర స్థిరరాశి uint32_t kmask1 = 0x03030303;
            స్థిర స్థిరరాశి uint32_t kmask2 = 0x0f0f0f0f;
            uint32_t auxs[4];
            జ్ఞాపకం_నకలు(auxs, x.scales, 12);
            స్థిరరాశి uint32_t tmp = auxs[2];
            auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            స్థిరరాశి int8_t * scales = (స్థిరరాశి int8_t *) &auxs[0];
            కొరకు (పూర్ణాంకం s = 0; s < NB; s++) { tile->scales[s_off + s] = (int32_t) scales[s] - 32; }
        }
    }
}

స్థిర శూన్య tiled_unpack_src0(స్థిరరాశి ఖండం_q2_K * అడ్డువరుసలు, int64_t అడ్డువరుస_stride, పూర్ణాంకం n_అడ్డువరుసలు, tiled_tile_src0 * tile, పూర్ణాంకం అంకె_k) {
    GGML_ASSERT(n_అడ్డువరుసలు <= TILED_TILE_అడ్డువరుసలు);
    constexpr పూర్ణాంకం NB = 16; // 16-wide subblocks
    స్థిరరాశి పూర్ణాంకం qk_stride = అంకె_k * TILED_TILE_K;
    స్థిరరాశి పూర్ణాంకం nb_stride = NB * అంకె_k;
    కొరకు (పూర్ణాంకం slab = 0; slab < అంకె_k; slab++) {
        కొరకు (పూర్ణాంకం r = 0; r < n_అడ్డువరుసలు; r++) {
            స్థిరరాశి పూర్ణాంకం d_off = slab * TILED_MICRO + r;
            స్థిరరాశి పూర్ణాంకం q_off = r * qk_stride + slab * TILED_TILE_K;
            స్థిరరాశి పూర్ణాంకం s_off = r * nb_stride + slab * NB;
            స్థిరరాశి ఖండం_q2_K & x = అడ్డువరుసలు[r * అడ్డువరుస_stride + slab];
            tile->d[d_off]    = ggml_fp16_గమ్యం_fp32(x.దత్తాంశం.దత్తాంశం.d);
            tile->dmin[d_off] = ggml_fp16_గమ్యం_fp32(x.దత్తాంశం.దత్తాంశం.dmin);

            // 2-bit code: element e -> half=e>>7, el=e&127
            //   byte = half*32 + (el & 31), shift = 2*(el >> 5)
            // see ggml_vec_dot_q2_K_q8_K_generic
            uint8_t * q = &tile->q[q_off];
            కొరకు (పూర్ణాంకం half = 0; half < 2; half++) {
                స్థిరరాశి uint8_t * s = x.qs + 32 * half;
                uint8_t * బహిర్గతం = q + 128 * half;
                tiled_unpk_2bit<0>(s, బహిర్గతం + 0);
                tiled_unpk_2bit<2>(s, బహిర్గతం + 32);
                tiled_unpk_2bit<4>(s, బహిర్గతం + 64);
                tiled_unpk_2bit<6>(s, బహిర్గతం + 96);
            }
            // scale/min packed in one byte per 16-element subblock: low 4 bits = scale, high 4 = min
            కొరకు (పూర్ణాంకం s = 0; s < NB; s++) {
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
స్థిర శూన్య tiled_unpack_src0(స్థిరరాశి ఖండం_iq4_xs * అడ్డువరుసలు, int64_t అడ్డువరుస_stride, పూర్ణాంకం n_అడ్డువరుసలు, tiled_tile_src0 * tile, పూర్ణాంకం అంకె_k) {
    GGML_ASSERT(n_అడ్డువరుసలు <= TILED_TILE_అడ్డువరుసలు);
    constexpr పూర్ణాంకం NB = 8; // 32-wide subblocks

    స్థిర స్థిరరాశి uint8_t lut[16] = {
        (uint8_t) (kvalues_iq4nl[0] + 128),  (uint8_t) (kvalues_iq4nl[1] + 128),
        (uint8_t) (kvalues_iq4nl[2] + 128),  (uint8_t) (kvalues_iq4nl[3] + 128),
        (uint8_t) (kvalues_iq4nl[4] + 128),  (uint8_t) (kvalues_iq4nl[5] + 128),
        (uint8_t) (kvalues_iq4nl[6] + 128),  (uint8_t) (kvalues_iq4nl[7] + 128),
        (uint8_t) (kvalues_iq4nl[8] + 128),  (uint8_t) (kvalues_iq4nl[9] + 128),
        (uint8_t) (kvalues_iq4nl[10] + 128), (uint8_t) (kvalues_iq4nl[11] + 128),
        (uint8_t) (kvalues_iq4nl[12] + 128), (uint8_t) (kvalues_iq4nl[13] + 128),
        (uint8_t) (kvalues_iq4nl[14] + 128), (uint8_t) (kvalues_iq4nl[15] + 128),
    };

    స్థిరరాశి పూర్ణాంకం qk_stride = అంకె_k * TILED_TILE_K;
    స్థిరరాశి పూర్ణాంకం nb_stride = NB * అంకె_k;
    కొరకు (పూర్ణాంకం slab = 0; slab < అంకె_k; slab++) {
        కొరకు (పూర్ణాంకం r = 0; r < n_అడ్డువరుసలు; r++) {
            స్థిరరాశి పూర్ణాంకం d_off = slab * TILED_MICRO + r;
            స్థిరరాశి పూర్ణాంకం q_off = r * qk_stride + slab * TILED_TILE_K;
            స్థిరరాశి పూర్ణాంకం s_off = r * nb_stride + slab * NB;
            స్థిరరాశి ఖండం_iq4_xs & x = అడ్డువరుసలు[r * అడ్డువరుస_stride + slab];
            tile->d[d_off] = ggml_fp16_గమ్యం_fp32(x.d);

            // 6-bit scale per 32, stored as (ls - 32); same extraction as dequantize_row_iq4_xs
            కొరకు (పూర్ణాంకం s = 0; s < NB; s++) {
                స్థిరరాశి పూర్ణాంకం ls = ((x.scales_l[s / 2] >> 4 * (s % 2)) & 0xf) | (((x.scales_h >> 2 * s) & 3) << 4);
                tile->scales[s_off + s] = (int32_t) ls - 32;
            }

            // 4-bit codes through the LUT: low nibbles of a byte pair come first
            uint8_t * q = &tile->q[q_off];
            uint8_t lo[32], hi[32];
            కొరకు (పూర్ణాంకం u = 0; u < 4; u++) {
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
స్థిర శూన్య tiled_unpack_src0(స్థిరరాశి ఖండం_iq2_xxs * అడ్డువరుసలు, int64_t అడ్డువరుస_stride, పూర్ణాంకం n_అడ్డువరుసలు, tiled_tile_src0 * tile, పూర్ణాంకం అంకె_k) {
    GGML_ASSERT(n_అడ్డువరుసలు <= TILED_TILE_అడ్డువరుసలు);
    constexpr పూర్ణాంకం NB = 8; // 32-wide subblocks

    స్థిరరాశి పూర్ణాంకం qk_stride = అంకె_k * TILED_TILE_K;
    స్థిరరాశి పూర్ణాంకం nb_stride = NB * అంకె_k;
    కొరకు (పూర్ణాంకం slab = 0; slab < అంకె_k; slab++) {
        కొరకు (పూర్ణాంకం r = 0; r < n_అడ్డువరుసలు; r++) {
            స్థిరరాశి పూర్ణాంకం d_off = slab * TILED_MICRO + r;
            స్థిరరాశి పూర్ణాంకం q_off = r * qk_stride + slab * TILED_TILE_K;
            స్థిరరాశి పూర్ణాంకం s_off = r * nb_stride + slab * NB;
            స్థిరరాశి ఖండం_iq2_xxs & x = అడ్డువరుసలు[r * అడ్డువరుస_stride + slab];
            tile->d[d_off] = ggml_fp16_గమ్యం_fp32(x.d) * 0.125f;

            uint32_t aux32[2];
            స్థిరరాశి uint8_t * aux8 = (స్థిరరాశి uint8_t *) aux32;
            uint64_t g[4];
            uint8_t signs4[4];
            కొరకు (పూర్ణాంకం ib32 = 0; ib32 < NB; ib32++) {
                జ్ఞాపకం_నకలు(aux32, x.qs + 4 * ib32, 2 * పరిమాణం(uint32_t));
                tile->scales[s_off + ib32] = (int32_t) (2 * (aux32[1] >> 28) + 1);
                కొరకు (పూర్ణాంకం l = 0; l < 4; l++) {
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
స్థిర శూన్య tiled_unpack_src0(స్థిరరాశి ఖండం_iq2_xs * అడ్డువరుసలు, int64_t అడ్డువరుస_stride, పూర్ణాంకం n_అడ్డువరుసలు, tiled_tile_src0 * tile, పూర్ణాంకం అంకె_k) {
    GGML_ASSERT(n_అడ్డువరుసలు <= TILED_TILE_అడ్డువరుసలు);
    constexpr పూర్ణాంకం NB = 16; // 16-wide subblocks

    స్థిరరాశి పూర్ణాంకం qk_stride = అంకె_k * TILED_TILE_K;
    స్థిరరాశి పూర్ణాంకం nb_stride = NB * అంకె_k;
    కొరకు (పూర్ణాంకం slab = 0; slab < అంకె_k; slab++) {
        కొరకు (పూర్ణాంకం r = 0; r < n_అడ్డువరుసలు; r++) {
            స్థిరరాశి పూర్ణాంకం d_off = slab * TILED_MICRO + r;
            స్థిరరాశి పూర్ణాంకం q_off = r * qk_stride + slab * TILED_TILE_K;
            స్థిరరాశి పూర్ణాంకం s_off = r * nb_stride + slab * NB;
            స్థిరరాశి ఖండం_iq2_xs & x = అడ్డువరుసలు[r * అడ్డువరుస_stride + slab];
            tile->d[d_off] = ggml_fp16_గమ్యం_fp32(x.d) * 0.125f;

            uint64_t g[4];
            uint8_t signs4[4];
            కొరకు (పూర్ణాంకం ib32 = 0; ib32 < QK_K / 32; ib32++) {
                tile->scales[s_off + 2 * ib32 + 0] = (int32_t) (2 * (x.scales[ib32] & 0xf) + 1);
                tile->scales[s_off + 2 * ib32 + 1] = (int32_t) (2 * (x.scales[ib32] >> 4) + 1);
                స్థిరరాశి uint16_t * q = x.qs + 4 * ib32;
                కొరకు (పూర్ణాంకం l = 0; l < 4; l++) {
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
స్థిర శూన్య tiled_unpack_src0(స్థిరరాశి ఖండం_iq2_s * అడ్డువరుసలు, int64_t అడ్డువరుస_stride, పూర్ణాంకం n_అడ్డువరుసలు, tiled_tile_src0 * tile, పూర్ణాంకం అంకె_k) {
    GGML_ASSERT(n_అడ్డువరుసలు <= TILED_TILE_అడ్డువరుసలు);
    constexpr పూర్ణాంకం NB = 16; // 16-wide subblocks

    స్థిరరాశి పూర్ణాంకం qk_stride = అంకె_k * TILED_TILE_K;
    స్థిరరాశి పూర్ణాంకం nb_stride = NB * అంకె_k;
    కొరకు (పూర్ణాంకం slab = 0; slab < అంకె_k; slab++) {
        కొరకు (పూర్ణాంకం r = 0; r < n_అడ్డువరుసలు; r++) {
            స్థిరరాశి పూర్ణాంకం d_off = slab * TILED_MICRO + r;
            స్థిరరాశి పూర్ణాంకం q_off = r * qk_stride + slab * TILED_TILE_K;
            స్థిరరాశి పూర్ణాంకం s_off = r * nb_stride + slab * NB;
            స్థిరరాశి ఖండం_iq2_s & x = అడ్డువరుసలు[r * అడ్డువరుస_stride + slab];
            tile->d[d_off] = ggml_fp16_గమ్యం_fp32(x.d) * 0.125f;

            స్థిరరాశి uint8_t * qs = x.qs;
            స్థిరరాశి uint8_t * signs = x.qs + QK_K / 8; // packed sign bytes share the qs array, same as the reference
            uint64_t g[4];
            కొరకు (పూర్ణాంకం ib32 = 0; ib32 < QK_K / 32; ib32++) {
                tile->scales[s_off + 2 * ib32 + 0] = (int32_t) (2 * (x.scales[ib32] & 0xf) + 1);
                tile->scales[s_off + 2 * ib32 + 1] = (int32_t) (2 * (x.scales[ib32] >> 4) + 1);
                కొరకు (పూర్ణాంకం l = 0; l < 4; l++) {
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
స్థిర శూన్య tiled_unpack_src0(స్థిరరాశి ఖండం_iq3_xxs * అడ్డువరుసలు, int64_t అడ్డువరుస_stride, పూర్ణాంకం n_అడ్డువరుసలు, tiled_tile_src0 * tile, పూర్ణాంకం అంకె_k) {
    GGML_ASSERT(n_అడ్డువరుసలు <= TILED_TILE_అడ్డువరుసలు);
    constexpr పూర్ణాంకం NB = 8; // 32-wide subblocks

    స్థిరరాశి పూర్ణాంకం qk_stride = అంకె_k * TILED_TILE_K;
    స్థిరరాశి పూర్ణాంకం nb_stride = NB * అంకె_k;
    కొరకు (పూర్ణాంకం slab = 0; slab < అంకె_k; slab++) {
        కొరకు (పూర్ణాంకం r = 0; r < n_అడ్డువరుసలు; r++) {
            స్థిరరాశి పూర్ణాంకం d_off = slab * TILED_MICRO + r;
            స్థిరరాశి పూర్ణాంకం q_off = r * qk_stride + slab * TILED_TILE_K;
            స్థిరరాశి పూర్ణాంకం s_off = r * nb_stride + slab * NB;
            స్థిరరాశి ఖండం_iq3_xxs & x = అడ్డువరుసలు[r * అడ్డువరుస_stride + slab];
            tile->d[d_off] = ggml_fp16_గమ్యం_fp32(x.d) * 0.25f;

            స్థిరరాశి uint8_t * qs = x.qs;
            స్థిరరాశి uint8_t * scales_and_signs = x.qs + QK_K / 4; // 4 bytes per 32: code bits in the top nibble, signs in 7-bit chunks
            uint64_t g[4];
            uint8_t signs4[4];
            కొరకు (పూర్ణాంకం ib32 = 0; ib32 < QK_K / 32; ib32++) {
                స్థిరరాశి uint32_t aux32 = *(స్థిరరాశి uint32_t *) (scales_and_signs + 4 * ib32);
                tile->scales[s_off + ib32] = (int32_t) (2 * (aux32 >> 28) + 1);
                కొరకు (పూర్ణాంకం l = 0; l < 4; l++) {
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
స్థిర శూన్య tiled_unpack_src0(స్థిరరాశి ఖండం_iq3_s * అడ్డువరుసలు, int64_t అడ్డువరుస_stride, పూర్ణాంకం n_అడ్డువరుసలు, tiled_tile_src0 * tile, పూర్ణాంకం అంకె_k) {
    GGML_ASSERT(n_అడ్డువరుసలు <= TILED_TILE_అడ్డువరుసలు);
    constexpr పూర్ణాంకం NB = 8; // 32-wide subblocks

    స్థిరరాశి పూర్ణాంకం qk_stride = అంకె_k * TILED_TILE_K;
    స్థిరరాశి పూర్ణాంకం nb_stride = NB * అంకె_k;
    కొరకు (పూర్ణాంకం slab = 0; slab < అంకె_k; slab++) {
        కొరకు (పూర్ణాంకం r = 0; r < n_అడ్డువరుసలు; r++) {
            స్థిరరాశి పూర్ణాంకం d_off = slab * TILED_MICRO + r;
            స్థిరరాశి పూర్ణాంకం q_off = r * qk_stride + slab * TILED_TILE_K;
            స్థిరరాశి పూర్ణాంకం s_off = r * nb_stride + slab * NB;
            స్థిరరాశి ఖండం_iq3_s & x = అడ్డువరుసలు[r * అడ్డువరుస_stride + slab];
            tile->d[d_off] = ggml_fp16_గమ్యం_fp32(x.d);

            స్థిరరాశి uint8_t * qs = x.qs;
            స్థిరరాశి uint8_t * qh = x.qh;
            స్థిరరాశి uint8_t * signs = x.signs;
            కొరకు (పూర్ణాంకం ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
                tile->scales[s_off + ib32 + 0] = (int32_t) (1 + 2 * (x.scales[ib32 / 2] & 0xf));
                tile->scales[s_off + ib32 + 1] = (int32_t) (1 + 2 * (x.scales[ib32 / 2] >> 4));
                కొరకు (పూర్ణాంకం h = 0; h < 2; h++) {
                    uint64_t g[4];
                    కొరకు (పూర్ణాంకం l = 0; l < 4; l++) {
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
స్థిర శూన్య tiled_unpack_src0(స్థిరరాశి ఖండం_iq1_s * అడ్డువరుసలు, int64_t అడ్డువరుస_stride, పూర్ణాంకం n_అడ్డువరుసలు, tiled_tile_src0 * tile, పూర్ణాంకం అంకె_k) {
    GGML_ASSERT(n_అడ్డువరుసలు <= TILED_TILE_అడ్డువరుసలు);
    constexpr పూర్ణాంకం NB = 8; // 32-wide subblocks

    స్థిరరాశి పూర్ణాంకం qk_stride = అంకె_k * TILED_TILE_K;
    స్థిరరాశి పూర్ణాంకం nb_stride = NB * అంకె_k;
    కొరకు (పూర్ణాంకం slab = 0; slab < అంకె_k; slab++) {
        కొరకు (పూర్ణాంకం r = 0; r < n_అడ్డువరుసలు; r++) {
            స్థిరరాశి పూర్ణాంకం d_off = slab * TILED_MICRO + r;
            స్థిరరాశి పూర్ణాంకం q_off = r * qk_stride + slab * TILED_TILE_K;
            స్థిరరాశి పూర్ణాంకం s_off = r * nb_stride + slab * NB;
            స్థిరరాశి ఖండం_iq1_s & x = అడ్డువరుసలు[r * అడ్డువరుస_stride + slab];
            tile->d[d_off] = ggml_fp16_గమ్యం_fp32(x.d) * 0.125f;

            స్థిరరాశి uint8_t * qs = x.qs;
            కొరకు (పూర్ణాంకం ib = 0; ib < QK_K / 32; ib++) {
                స్థిరరాశి uint16_t hw = x.qh[ib];
                tile->scales[s_off + ib] = (int32_t) (2 * ((hw >> 12) & 7) + 1);
                స్థిరరాశి int8_t delta = (hw & 0x8000) ? -1 : 1;
                కొరకు (పూర్ణాంకం l = 0; l < 4; l++) {
                    స్థిరరాశి uint64_t నమోదు_2 = iq1s_grid[qs[l] | (((hw >> (3 * l)) & 7) << 8)];
                    tiled_unpk_tern8((స్థిరరాశి uint8_t *) &నమోదు_2, delta, &tile->q[q_off + 32 * ib + 8 * l]);
                }
                qs += 4;
            }
        }
    }
}

// iq1_m: like iq1_s but the fp16 scale is packed across the 4 scale bytes (no d field) and the
// delta offset is per 8, giving one scale per 16; 3-bit scale per 16, no min
// codes stored as (8 * grid + delta + 128), matching BIAS = 128
స్థిర శూన్య tiled_unpack_src0(స్థిరరాశి ఖండం_iq1_m * అడ్డువరుసలు, int64_t అడ్డువరుస_stride, పూర్ణాంకం n_అడ్డువరుసలు, tiled_tile_src0 * tile, పూర్ణాంకం అంకె_k) {
    GGML_ASSERT(n_అడ్డువరుసలు <= TILED_TILE_అడ్డువరుసలు);
    constexpr పూర్ణాంకం NB = 16; // 16-wide subblocks

    స్థిరరాశి పూర్ణాంకం qk_stride = అంకె_k * TILED_TILE_K;
    స్థిరరాశి పూర్ణాంకం nb_stride = NB * అంకె_k;
    కొరకు (పూర్ణాంకం slab = 0; slab < అంకె_k; slab++) {
        కొరకు (పూర్ణాంకం r = 0; r < n_అడ్డువరుసలు; r++) {
            స్థిరరాశి పూర్ణాంకం d_off = slab * TILED_MICRO + r;
            స్థిరరాశి పూర్ణాంకం q_off = r * qk_stride + slab * TILED_TILE_K;
            స్థిరరాశి పూర్ణాంకం s_off = r * nb_stride + slab * NB;
            స్థిరరాశి ఖండం_iq1_m & x = అడ్డువరుసలు[r * అడ్డువరుస_stride + slab];

            స్థిరరాశి uint16_t * sc = (స్థిరరాశి uint16_t *) x.scales;
            iq1m_scale_t scale;
            scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
            tile->d[d_off] = ggml_fp16_గమ్యం_fp32(scale.f16) * 0.125f;

            స్థిరరాశి uint8_t * qs = x.qs;
            స్థిరరాశి uint8_t * qh = x.qh;
            కొరకు (పూర్ణాంకం ib = 0; ib < QK_K / 32; ib++) {
                స్థిరరాశి uint16_t hw = sc[ib / 2];
                స్థిరరాశి పూర్ణాంకం sh = 6 * (ib % 2);
                tile->scales[s_off + 2 * ib + 0] = (int32_t) (2 * ((hw >> sh) & 7) + 1);
                tile->scales[s_off + 2 * ib + 1] = (int32_t) (2 * ((hw >> (sh + 3)) & 7) + 1);

                స్థిరరాశి uint16_t idx[4] = {
                    (uint16_t) (qs[0] | ((qh[0] << 8) & 0x700)),
                    (uint16_t) (qs[1] | ((qh[0] << 4) & 0x700)),
                    (uint16_t) (qs[2] | ((qh[1] << 8) & 0x700)),
                    (uint16_t) (qs[3] | ((qh[1] << 4) & 0x700)),
                };
                స్థిరరాశి int8_t delta[4] = {
                    (int8_t) ((qh[0] & 0x08) ? -1 : 1), (int8_t) ((qh[0] & 0x80) ? -1 : 1),
                    (int8_t) ((qh[1] & 0x08) ? -1 : 1), (int8_t) ((qh[1] & 0x80) ? -1 : 1),
                };
                కొరకు (పూర్ణాంకం l = 0; l < 4; l++) {
                    స్థిరరాశి uint64_t నమోదు_2 = iq1s_grid[idx[l]];
                    tiled_unpk_tern8((స్థిరరాశి uint8_t *) &నమోదు_2, delta[l], &tile->q[q_off + 32 * ib + 8 * l]);
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
స్థిర శూన్య tiled_unpack_src1_q8_K(స్థిరరాశి ఖండం_q8_K * స్థిరరాశి * అడ్డువరుసలు, పూర్ణాంకం n_అడ్డువరుసలు, tiled_tile_src1 * tile,
                                   పూర్ణాంకం kblk, పూర్ణాంకం అంకె_k) {
    GGML_ASSERT(n_అడ్డువరుసలు <= TILED_TILE_అడ్డువరుసలు);
    స్థిరరాశి పూర్ణాంకం n_padded = (n_అడ్డువరుసలు + TILED_MICRO - 1) & ~(TILED_MICRO - 1);
    స్థిరరాశి పూర్ణాంకం bs_stride = (అంకె_k == 1) ? TILED_TILE_అడ్డువరుసలు : TILED_MICRO;
    స్థిరరాశి పూర్ణాంకం n16 = TILED_TILE_K / TILED_MICRO;
    // natural [row][k] fill, zero-pad ragged tail; codes stay natural here, the driver
    // repacks them just-in-time via tiled_repack_src1 (per 16-row band on the standard path,
    // the whole chunk on the narrow path) so each repacked region is L1-hot for its uses
    కొరకు (పూర్ణాంకం slab = 0; slab < అంకె_k; slab++) {
        స్థిరరాశి పూర్ణాంకం q_off = slab * TILED_TILE_K;
        కొరకు (పూర్ణాంకం r = 0; r < n_padded; r++) {
            ఒకవేళ (r < n_అడ్డువరుసలు) {
                జ్ఞాపకం_నకలు(&tile->q[r * (అంకె_k * TILED_TILE_K) + q_off], అడ్డువరుసలు[r][kblk + slab].qs, TILED_TILE_K);
            } లేకపోతే {
                memset(&tile->q[r * (అంకె_k * TILED_TILE_K) + q_off], 0, TILED_TILE_K);
            }
        }
        // d and bsums (ISA-independent)
        కొరకు (పూర్ణాంకం r = 0; r < n_padded; r++) {
            ఒకవేళ (r < n_అడ్డువరుసలు) {
                స్థిరరాశి ఖండం_q8_K & x = అడ్డువరుసలు[r][kblk + slab];
                కొరకు (పూర్ణాంకం s = 0; s < n16; s++) { tile->bsums[(slab * n16 + s) * bs_stride + r] = (int32_t) x.bsums[s]; }
                tile->d[slab * TILED_MICRO + r] = x.d;
            } లేకపోతే {
                కొరకు (పూర్ణాంకం s = 0; s < n16; s++) { tile->bsums[(slab * n16 + s) * bs_stride + r] = 0; }
                tile->d[slab * TILED_MICRO + r] = 0.0f;
            }
        }
    }
}

// GGML_CPU_TILED_MM: master switch, on by default. If off, we fast return false and normal vec_dot mul_mat resumes
స్థిర తార్కిక ggml_tiled_matmul_enabled(శూన్య) {
    స్థిర తార్కిక enabled = నిజం;
    స్థిర std::once_flag flag;
    std::call_once(flag, []() {
        స్థిరరాశి అక్షరం * env = getenv("GGML_CPU_TILED_MM");
        enabled = env == ఖాళీ || atoi(env) != 0;
    });
    తిరిగి enabled;
}

// GGML_CPU_TILED_MM_FORCE: test/bench only, take the tiled path even when unprofitable
స్థిర తార్కిక ggml_tiled_matmul_forced(శూన్య) {
    స్థిర తార్కిక forced = అబద్ధం;
    స్థిర std::once_flag flag;
    std::call_once(flag, []() {
        స్థిరరాశి అక్షరం * env = getenv("GGML_CPU_TILED_MM_FORCE");
        forced = env != ఖాళీ && atoi(env) == 1;
    });
    తిరిగి forced;
}

// hard constraints shared by the MUL_MAT and MUL_MAT_ID entries; the src0 type gate is the
// entries' type switch

#ఒకవేళ !defined(__AVX512VNNI__) && !defined(__AVX2__) && !defined(__AVX__)
స్థిర తార్కిక ggml_tiled_supported(స్థిరరాశి నిర్మాణం ggml_tensor * src0,
                                 స్థిరరాశి నిర్మాణం ggml_tensor * src1) {
    వాడని(src0);
    వాడని(src1);
    తిరిగి అబద్ధం;
}
#లేకపోతే
స్థిర తార్కిక ggml_tiled_supported(స్థిరరాశి నిర్మాణం ggml_tensor * src0,
                                 స్థిరరాశి నిర్మాణం ggml_tensor * src1) {
    ఒకవేళ (!ggml_tiled_matmul_enabled()) {
        తిరిగి అబద్ధం;
    }

    // repack-buffer weights hold a repacked layout, let that kernel handle
    ఒకవేళ (src0->extra != ఖాళీ) {
        తిరిగి అబద్ధం;
    }
    // Supported quant types for src0
    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_Q6_K:
        స్థితి GGML_TYPE_Q5_K:
        స్థితి GGML_TYPE_Q4_K:
        స్థితి GGML_TYPE_Q3_K:
        స్థితి GGML_TYPE_Q2_K:
        స్థితి GGML_TYPE_IQ4_XS:
        స్థితి GGML_TYPE_IQ2_XXS:
        స్థితి GGML_TYPE_IQ2_XS:
        స్థితి GGML_TYPE_IQ2_S:
        స్థితి GGML_TYPE_IQ3_XXS:
        స్థితి GGML_TYPE_IQ3_S:
        స్థితి GGML_TYPE_IQ1_S:
        స్థితి GGML_TYPE_IQ1_M:
            తిరిగి నిజం;
        అప్రమేయ:
            తిరిగి అబద్ధం;
    }
    ఒకవేళ (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_Q8_K) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (src1->type == GGML_TYPE_Q8_K && !ggml_is_contiguous(src1)) {
        // We can handle noncontiguous floats because we're repacking to q8_k anyways
        తిరిగి అబద్ధం;
    }
    తిరిగి నిజం;
}
#అయితే_ముగింపు

// per-thread workspace slot size (0 when tiled is disabled or unsupported on this arch)
స్థిర size_t ggml_tiled_ws_కొలత(శూన్య) {
    ఒకవేళ (!ggml_tiled_matmul_enabled()) {
        తిరిగి 0;
    }
    తిరిగి TILED_WS_స్థానం; // clean 512KB slot, rounded up from sizeof(tiled_ws)
}

size_t ggml_tiled_wdata_కొలత(పూర్ణాంకం n_tasks, నిర్మాణం ggml_tensor * dst) {
    ఒకవేళ (! ggml_tiled_supported(dst->src[0], dst->src[1])) {
        తిరిగి 0; // unsupported, don't allocate
    }
    తిరిగి 64 + n_tasks * ggml_tiled_ws_కొలత();  // 64 for alignment plus one 512KB slot per thread
}


// narrow-path K chunk: how many K's worth of weights to read contiguously into long/skinny tiles
// Assumes 32kb L1 cache budget
స్థిర పూర్ణాంకం ggml_tiled_narrow_k_extent(int64_t n_అడ్డువరుసలు, int64_t ne00) {
    పూర్ణాంకం l1 = 31 * 1024;  // fit weights in 31k to leave room for scales

    పూర్ణాంకం అడ్డువరుసలు = (పూర్ణాంకం) n_అడ్డువరుసలు;
    ఒకవేళ (అడ్డువరుసలు < TILED_MICRO) అడ్డువరుసలు = TILED_MICRO;
    పూర్ణాంకం ke = l1 / (TILED_MICRO + అడ్డువరుసలు);
    ke &= ~(TILED_TILE_K - 1);  // floor to a multiple of 256
    స్థిరరాశి పూర్ణాంకం ke_max = TILED_TILE_అడ్డువరుసలు * TILED_TILE_K / TILED_MICRO; // buffer ceiling
    ఒకవేళ (ke > ke_max) ke = ke_max;
    అయితే (ke >= TILED_TILE_K && ne00 % ke != 0) {
        ke -= TILED_TILE_K;
    }
    తిరిగి ke < TILED_TILE_K ? 0 : ke;
}

// Writeback of the 256x256 window: buf is j-major (row stride buf_stride), dst is i-major (column stride dst_stride).
స్థిర శూన్య tiled_store_కిటికీ(స్థిరరాశి దశాంశం * buf, పూర్ణాంకం n_src0, పూర్ణాంకం n_src1, పూర్ణాంకం buf_stride,
                               దశాంశం * dst, size_t dst_stride) {
    పూర్ణాంకం ri = 0;
    కొరకు (; ri + 16 <= n_src0; ri += 16) {
        పూర్ణాంకం rj = 0;
        కొరకు (; rj + 8 <= n_src1; rj += 8) {
            దశాంశం r[16][8];
            కొరకు (పూర్ణాంకం t = 0; t < 16; t++) {
                కొరకు (పూర్ణాంకం u = 0; u < 8; u++) {
                    r[t][u] = buf[(ri + t) * buf_stride + rj + u];
                }
            }
            కొరకు (పూర్ణాంకం u = 0; u < 8; u++) {
                కొరకు (పూర్ణాంకం t = 0; t < 16; t++) {
                    dst[(ri + t) + (size_t) (rj + u) * dst_stride] = r[t][u];
                }
            }
        }
        // ragged j tail
        కొరకు (; rj < n_src1; rj++) {
            కొరకు (పూర్ణాంకం t = 0; t < 16; t++) {
                dst[(ri + t) + (size_t) rj * dst_stride] = buf[(ri + t) * buf_stride + rj];
            }
        }
    }
    // ragged i tail
    కొరకు (; ri < n_src0; ri++) {
        కొరకు (పూర్ణాంకం j = 0; j < n_src1; j++) {
            dst[ri + (size_t) j * dst_stride] = buf[ri * buf_stride + j];
        }
    }
}

// MUL_MAT_ID (MoE): src1 rows are gathered per output row (expert dispatch) and dst rows are
// scattered back. The expert's cne1 gathered q8_K rows are staged once per expert into
// thread-local scratch (contiguous rows, plus the [k/4][row][4] interleave on VNNI) so the
// unpack and microtile are reused unchanged from mul_mat.

// src0 rows per g group; finer than the TILED_TILE_K kernel tile so all threads stay busy at small ne01
#నిర్వచించు TILED_MMID_సమూహం 64

// like tiled_store_window, but the dst columns are not contiguous: col_ptrs[j] points at the
// start of dst column j, whose rows are contiguous (dim 0, nb0 == 4 bytes).
స్థిర శూన్య tiled_store_కిటికీ_scatter(స్థిరరాశి దశాంశం * buf, పూర్ణాంకం n_src0, పూర్ణాంకం n_src1, పూర్ణాంకం buf_stride,
                                       దశాంశం * స్థిరరాశి * నిలువు_ptrs) {
    పూర్ణాంకం ri = 0;
    కొరకు (; ri + 16 <= n_src0; ri += 16) {
        పూర్ణాంకం rj = 0;
        కొరకు (; rj + 8 <= n_src1; rj += 8) {
            దశాంశం r[16][8];
            కొరకు (పూర్ణాంకం t = 0; t < 16; t++) {
                కొరకు (పూర్ణాంకం u = 0; u < 8; u++) {
                    r[t][u] = buf[(ri + t) * buf_stride + rj + u];
                }
            }
            కొరకు (పూర్ణాంకం u = 0; u < 8; u++) {
                కొరకు (పూర్ణాంకం t = 0; t < 16; t++) {
                    నిలువు_ptrs[rj + u][ri + t] = r[t][u];
                }
            }
        }
        // ragged j tail
        కొరకు (; rj < n_src1; rj++) {
            కొరకు (పూర్ణాంకం t = 0; t < 16; t++) {
                నిలువు_ptrs[rj][ri + t] = buf[(ri + t) * buf_stride + rj];
            }
        }
    }
    // ragged i tail
    కొరకు (; ri < n_src0; ri++) {
        కొరకు (పూర్ణాంకం j = 0; j < n_src1; j++) {
            నిలువు_ptrs[j][ri] = buf[ri * buf_stride + j];
        }
    }
}


// one (g, k) macrotile: zero the acc window, sweep K in 256 element slabs, scatter the result rows
// rows points at this k window's routed src1 rows, row r at its base block
మూస <రకం_పేరు B, పూర్ణాంకం SUBBLK, తార్కిక HAS_MIN, పూర్ణాంకం BIAS, తార్కిక ACTBIAS>
స్థిర శూన్య tiled_mmid_gemm_కిటికీ(నిర్మాణం ggml_tensor * dst, స్థిరరాశి నిర్మాణం ggml_tensor * src0,
                                   స్థిరరాశి అక్షరం * src0_cur, int64_t r, int64_t k, int64_t nrows,
                                   స్థిరరాశి int32_t * expert_అడ్డువరుసలు,
                                   స్థిరరాశి ఖండం_q8_K * స్థిరరాశి * అడ్డువరుసలు,
                                   tiled_ws * ws) {
    స్థిరరాశి int64_t ne00 = src0->ne[0];
    స్థిరరాశి int64_t ne01 = src0->ne[1];

    స్థిరరాశి int64_t r_చివర = MIN(r + TILED_MMID_సమూహం, ne01);
    స్థిరరాశి పూర్ణాంకం n_src0 = (పూర్ణాంకం) (r_చివర - r);

    స్థిరరాశి size_t src0_bs = ggml_type_కొలత(src0->type);
    స్థిరరాశి int64_t src0_stride = src0->nb[1] / src0_bs;

    // the window is at most TILED_MMID_GROUP x TILED_TILE_K, so zero only that region of acc
    కొరకు (int64_t i = 0; i < n_src0; i++) {
        memset(&ws->acc[i * TILED_TILE_అడ్డువరుసలు], 0, nrows * పరిమాణం(దశాంశం));
    }

    // scattered writeback: column m goes to its routed dst row; r * nb[0] is the window row offset
    దశాంశం * నిలువు_ptrs[TILED_TILE_అడ్డువరుసలు];
    కొరకు (int64_t m = 0; m < nrows; m++) {
        నిలువు_ptrs[m] = (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + r * dst->nb[0] + expert_అడ్డువరుసలు[2 * (k + m) + 0] * dst->nb[1] +
                                 expert_అడ్డువరుసలు[2 * (k + m) + 1] * dst->nb[2]);
    }

    // narrow path (small nrows): we're memory bound in this case so do longer, skinny tiled in order
    // to have contiguous reads and improve memory bandwidth
    ఒకవేళ (nrows <= TILED_MICRO) {
        స్థిరరాశి పూర్ణాంకం k_extent = ggml_tiled_narrow_k_extent(nrows, ne00);
        స్థిరరాశి పూర్ణాంకం అంకె_k = k_extent / TILED_TILE_K;
        కొరకు (int64_t k0 = 0; k0 < ne00; k0 += k_extent) {
            స్థిరరాశి పూర్ణాంకం kstart = (పూర్ణాంకం) (k0 / TILED_TILE_K);
            // activation: one 16-row band (nrows <= 16). Unpack the whole k_extent chunk (all
            // slabs) so each activation row is a long stream, then repack each slab in place
            tiled_unpack_src1_q8_K(అడ్డువరుసలు, nrows, &ws->src1, kstart, అంకె_k);
            // repack the whole k_extent chunk in place: num_k slabs x 4 tiles, 16 rows, row stride num_k*256
            tiled_repack_src1(&ws->src1, 0, అంకె_k, ACTBIAS);
            // weight groups (16 at a time): unpack the k_extent chunk of the group (the long per-row
            // read), then one standard MAC per slab accumulating into the same acc rows
            కొరకు (int64_t ir0 = r; ir0 < r_చివర; ir0 += TILED_MICRO) {
                స్థిరరాశి పూర్ణాంకం n0 = (పూర్ణాంకం) MIN(TILED_MICRO, r_చివర - ir0);
                స్థిరరాశి B * wbase = (స్థిరరాశి B *) (src0_cur + ir0 * src0->nb[1] + kstart * src0_bs);
                tiled_unpack_src0(wbase, src0_stride, n0, &ws->src0, అంకె_k);
                tiled_repack_src0<SUBBLK>(&ws->src0, n0, అంకె_k, BIAS, ACTBIAS);
                దశాంశం * buf = ws->acc + (ir0 - r) * TILED_TILE_అడ్డువరుసలు;
                కొరకు (పూర్ణాంకం slab = 0; slab < అంకె_k; slab++) {
                    tiled_నిరంతరం_microtile<SUBBLK, HAS_MIN, BIAS, ACTBIAS>(ws->src0, ws->src1, 0, 0, అంకె_k, slab, buf, TILED_TILE_అడ్డువరుసలు);
                }
            }
        }
    } లేకపోతే {
        // standard per-slab path: K is stepped in 256-element slabs
        కొరకు (int64_t ib = 0; ib < ne00; ib += TILED_TILE_K) {
            స్థిరరాశి పూర్ణాంకం kblk = (పూర్ణాంకం) (ib / TILED_TILE_K);
            tiled_unpack_src0((స్థిరరాశి B *) (src0_cur + r * src0->nb[1] + kblk * src0_bs), src0_stride, n_src0, &ws->src0, 1);
            tiled_repack_src0<SUBBLK>(&ws->src0, n_src0, 1, BIAS, ACTBIAS);
            tiled_unpack_src1_q8_K(అడ్డువరుసలు, nrows, &ws->src1, kblk, 1);
            // 16x16 microtiles sweeping the window; repack each src1 band just-in-time
            // (j0-outer) so only the bands actually used are repacked and each is L1-hot
            కొరకు (int64_t ir1 = 0; ir1 < nrows; ir1 += TILED_MICRO) {
                tiled_repack_src1(&ws->src1, (పూర్ణాంకం) ir1, 1, ACTBIAS);
                కొరకు (int64_t ir0 = r; ir0 < r_చివర; ir0 += TILED_MICRO) {
                    tiled_నిరంతరం_microtile<SUBBLK, HAS_MIN, BIAS, ACTBIAS>(ws->src0, ws->src1,
                        (పూర్ణాంకం) (ir0 - r), (పూర్ణాంకం) ir1, 1, 0,
                        ws->acc, TILED_TILE_అడ్డువరుసలు);
                }
            }
        }
    }

    tiled_store_కిటికీ_scatter(ws->acc, n_src0, nrows, TILED_TILE_అడ్డువరుసలు, నిలువు_ptrs);
}

// one expert of MUL_MAT_ID: each k window's dispatched rows are pointed at by a per-window
// row pointer list and swept over the thread's row windows, the dst rows are scattered back
// per window
మూస <రకం_పేరు B, పూర్ణాంకం SUBBLK, తార్కిక HAS_MIN, పూర్ణాంకం BIAS, తార్కిక ACTBIAS>
స్థిర శూన్య ggml_compute_forward_mul_mat_id_tiled_one_expert(
        స్థిరరాశి నిర్మాణం ggml_compute_params * params,
              నిర్మాణం ggml_tensor *         dst,
        int64_t                            cur_a,
        int64_t                            cne1,
        స్థిరరాశి int32_t *                    expert_అడ్డువరుసలు,
        అక్షరం *                             scratch) {

    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి గణన ggml_type vec_dot_type = ggml_get_type_traits_cpu(src0->type)->vec_dot_type;

    స్థిరరాశి size_t nbw1 = (src1->type == vec_dot_type) ? nb11 : ggml_అడ్డువరుస_కొలత(vec_dot_type, ne10);
    స్థిరరాశి శూన్య * wdata = (src1->type == vec_dot_type) ? src1->దత్తాంశం : params->wdata;

    స్థిరరాశి అక్షరం * src0_cur = (స్థిరరాశి అక్షరం *) src0->దత్తాంశం + cur_a * nb02;

    // this thread's workspace
    tiled_ws * ws = (tiled_ws *) (scratch + (size_t) ith * ggml_tiled_ws_కొలత());

    // groups of TILED_MMID_GROUP rows; rounded up, the window tail is clamped in the gemm
    స్థిరరాశి int64_t ngroups = (ne01 + TILED_MMID_సమూహం - 1) / TILED_MMID_సమూహం;

    స్థిరరాశి int64_t g0 = (ngroups * ith) / nth;
    స్థిరరాశి int64_t g1 = (ngroups * (ith + 1)) / nth;

    // no rows for this thread; nothing to do
    ఒకవేళ (g0 >= g1) {
        తిరిగి;
    }

    // MUL_MAT_ID scatters the src1 rows per routed row: point each k window's rows
    // at the routed src1 rows, then sweep the row windows
    కొరకు (int64_t k = 0; k < cne1; k += TILED_TILE_K) {
        స్థిరరాశి int64_t nrows = MIN(TILED_TILE_K, cne1 - k);

        స్థిరరాశి ఖండం_q8_K * అడ్డువరుసలు[TILED_TILE_అడ్డువరుసలు];
        కొరకు (int64_t i = 0; i < nrows; i++) {
            స్థిరరాశి int64_t i11 = expert_అడ్డువరుసలు[2 * (k + i) + 0] % ne11;
            స్థిరరాశి int64_t i12 = expert_అడ్డువరుసలు[2 * (k + i) + 1];
            అడ్డువరుసలు[i] = (స్థిరరాశి ఖండం_q8_K *) ((స్థిరరాశి అక్షరం *) wdata + (i11 + i12 * ne11) * nbw1);
        }

        కొరకు (int64_t g = g0; g < g1; g++) {
            స్థిరరాశి int64_t r = g * TILED_MMID_సమూహం;

            tiled_mmid_gemm_కిటికీ<B, SUBBLK, HAS_MIN, BIAS, ACTBIAS>(dst, src0, src0_cur, r, k, nrows, expert_అడ్డువరుసలు, అడ్డువరుసలు, ws);
        }
    }
}

మూస <రకం_పేరు B, పూర్ణాంకం SUBBLK, తార్కిక HAS_MIN, పూర్ణాంకం BIAS, తార్కిక ACTBIAS>
స్థిర శూన్య ggml_compute_forward_mul_mat_tiled_one_chunk(
    స్థిరరాశి నిర్మాణం ggml_compute_params * params,
    నిర్మాణం ggml_tensor * dst,
    స్థిరరాశి int64_t ir0_ప్రారంభం,
    స్థిరరాశి int64_t ir0_చివర,
    స్థిరరాశి int64_t ir1_ప్రారంభం,
    స్థిరరాశి int64_t ir1_చివర,
    tiled_ws * ws) {

    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి గణన ggml_type vec_dot_type = ggml_get_type_traits_cpu(src0->type)->vec_dot_type;

    // broadcast factors
    స్థిరరాశి int64_t r2 = ne12 / ne02;
    స్థిరరాశి int64_t r3 = ne13 / ne03;

    ఒకవేళ (ir0_ప్రారంభం >= ir0_చివర || ir1_ప్రారంభం >= ir1_చివర) {
        తిరిగి;
    }

    స్థిరరాశి శూన్య * wdata = (src1->type == vec_dot_type) ? src1->దత్తాంశం : params->wdata;
    స్థిరరాశి size_t అడ్డువరుస_కొలత = ggml_అడ్డువరుస_కొలత(vec_dot_type, ne10);
    స్థిరరాశి size_t src0_bs  = ggml_type_కొలత(src0->type);
    స్థిరరాశి size_t src1_bs  = ggml_type_కొలత(vec_dot_type);

    GGML_ASSERT(ne00 % 256 == 0);
    ప్రతిజ్ఞా(ne12 % ne02 == 0);
    ప్రతిజ్ఞా(ne13 % ne03 == 0);

    స్థిరరాశి int64_t src0_stride = nb01 / src0_bs;  // blocks between src0 rows
    స్థిరరాశి int64_t src1_stride = (src1->type == vec_dot_type ? src1->nb[1] : అడ్డువరుస_కొలత) / src1_bs;

    స్థిరరాశి int64_t TILE = 256;
    స్థిరరాశి int64_t MICRO = 16;

    // 256-wide windows over the chunk. The iir1 window is additionally clamped at the
    // src1 batch (ne11) boundary: the tiles require a constant batch index (i12/i13)
    // within a window, so advance by the clamped end, not a fixed 256.
    కొరకు (int64_t iir1 = ir1_ప్రారంభం; iir1 < ir1_చివర; ) {
        int64_t iir1_చివర = MIN(iir1 + TILE, ir1_చివర);
        స్థిరరాశి int64_t bnd = (iir1 / ne11 + 1) * ne11;
        ఒకవేళ (bnd < iir1_చివర) {
            iir1_చివర = bnd;
        }

        స్థిరరాశి పూర్ణాంకం n_src1 = (పూర్ణాంకం) (iir1_చివర - iir1);

        // batch coords, constant within the clamped window
        స్థిరరాశి int64_t i13 = iir1 / (ne12 * ne11);
        స్థిరరాశి int64_t i12 = (iir1 - i13 * ne12 * ne11) / ne11;
        // within-batch row; dst_col below holds the i12/i13 batch offset, so the store
        // applies i11 * nb1 (not the flattened iir1, which spans all batch dims)
        స్థిరరాశి int64_t i11 = iir1 - i13 * ne12 * ne11 - i12 * ne11;

        // dst batches == src1 batches (ggml_mul_mat), so the loop batch coords are in
        // src1 space; src0 batches are broadcast over them, map down into src0's batch
        స్థిరరాశి int64_t i02 = i12 / r2;
        స్థిరరాశి int64_t i03 = i13 / r3;

        స్థిరరాశి అక్షరం * src0_అడ్డువరుస = (స్థిరరాశి అక్షరం *) src0->దత్తాంశం + i02 * src0->nb[2] + i03 * src0->nb[3];
        అక్షరం * dst_నిలువు = (అక్షరం *) dst->దత్తాంశం + i12 * nb2 + i13 * nb3;

        // rows[r] is the window's row r base block; the k-slab offset is applied in the unpack
        స్థిరరాశి ఖండం_q8_K * అడ్డువరుసలు[TILED_TILE_అడ్డువరుసలు];
        కొరకు (పూర్ణాంకం r = 0; r < n_src1; r++) {
            అడ్డువరుసలు[r] = (స్థిరరాశి ఖండం_q8_K *) ((స్థిరరాశి అక్షరం *) wdata + (iir1 + r) * src1_stride * src1_bs);
        }

        కొరకు (int64_t iir0 = ir0_ప్రారంభం; iir0 < ir0_చివర; iir0 += TILE) {
            int64_t iir0_చివర = MIN(iir0 + TILE, ir0_చివర);
            స్థిరరాశి పూర్ణాంకం n_src0 = (పూర్ణాంకం) (iir0_చివర - iir0);

            // result buffer zeroed once per macrotile
            memset(ws->acc, 0, (size_t)TILED_TILE_అడ్డువరుసలు * TILED_TILE_అడ్డువరుసలు * పరిమాణం(దశాంశం));

            ఒకవేళ (n_src1 <= TILED_MICRO) {
                // narrow path: if we have few enough rows, we're memory bound
                // Do longer, skinner tiles so we have longer contiguous reads and improve bandwidth
                స్థిరరాశి పూర్ణాంకం k_extent = ggml_tiled_narrow_k_extent(n_src1, ne00);
                స్థిరరాశి పూర్ణాంకం అంకె_k = k_extent / TILED_TILE_K;
                కొరకు (int64_t k0 = 0; k0 < ne00; k0 += k_extent) {
                    స్థిరరాశి పూర్ణాంకం kstart = (పూర్ణాంకం) (k0 / TILED_TILE_K);
                    // activation: one 16-row band (n_src1 <= 16). Unpack the whole k_extent chunk
                    // (all slabs) so each activation row is a long stream, then repack each slab in place
                    tiled_unpack_src1_q8_K(అడ్డువరుసలు, n_src1, &ws->src1, kstart, అంకె_k);
                    // repack the whole k_extent chunk in place, if kernel wants to: num_k slabs x 4 tiles, 16 rows, row stride num_k*256
                    tiled_repack_src1(&ws->src1, 0, అంకె_k, ACTBIAS);
                    // weight groups: unpack the k_extent chunk of each 16-row group (the long per-row
                    // read), then one standard MAC per slab accumulating into the same acc rows
                    కొరకు (int64_t ir0 = iir0; ir0 < iir0_చివర; ir0 += MICRO) {
                        స్థిరరాశి పూర్ణాంకం n0 = (పూర్ణాంకం) MIN(MICRO, iir0_చివర - ir0);
                        స్థిరరాశి B * wbase = (స్థిరరాశి B *) (src0_అడ్డువరుస + ir0 * nb01 + kstart * src0_bs);
                        tiled_unpack_src0(wbase, src0_stride, n0, &ws->src0, అంకె_k);
                        tiled_repack_src0<SUBBLK>(&ws->src0, n0, అంకె_k, BIAS, ACTBIAS);
                        దశాంశం * buf = ws->acc + (ir0 - iir0) * TILED_TILE_అడ్డువరుసలు;
                        కొరకు (పూర్ణాంకం slab = 0; slab < అంకె_k; slab++) {
                            tiled_నిరంతరం_microtile<SUBBLK, HAS_MIN, BIAS, ACTBIAS>(ws->src0, ws->src1, 0, 0, అంకె_k, slab, buf, TILED_TILE_అడ్డువరుసలు);
                        }
                    }
                }
            } లేకపోతే {
                // standard per-slab path: K is stepped in 256-element chunks
                కొరకు (int64_t ib = 0; ib < ne00; ib += TILE) {
                    స్థిరరాశి పూర్ణాంకం kblk = (పూర్ణాంకం) (ib / TILE);
                    tiled_unpack_src0((స్థిరరాశి B *) (src0_అడ్డువరుస + iir0 * nb01 + kblk * src0_bs), src0_stride, n_src0, &ws->src0, 1);
                    tiled_repack_src0<SUBBLK>(&ws->src0, n_src0, 1, BIAS, ACTBIAS);
                    tiled_unpack_src1_q8_K(అడ్డువరుసలు, n_src1, &ws->src1, kblk, 1);

                    // 16x16 microtiles sweeping the window; repack each src1 band before first use
                    కొరకు (int64_t ir1 = iir1; ir1 < iir1_చివర; ir1 += MICRO) {
                        tiled_repack_src1(&ws->src1, (పూర్ణాంకం) (ir1 - iir1), 1, ACTBIAS);
                        కొరకు (int64_t ir0 = iir0; ir0 < iir0_చివర; ir0 += MICRO) {
                            tiled_నిరంతరం_microtile<SUBBLK, HAS_MIN, BIAS, ACTBIAS>(ws->src0, ws->src1,
                                (పూర్ణాంకం) (ir0 - iir0), (పూర్ణాంకం) (ir1 - iir1), 1, 0,
                                ws->acc, TILED_TILE_అడ్డువరుసలు);
                        }
                    }
                }
            }
            // write acc back out from L2 to main memory
            tiled_store_కిటికీ(ws->acc, n_src0, n_src1, TILED_TILE_అడ్డువరుసలు,
                               (దశాంశం *) (dst_నిలువు + iir0 * nb0 + i11 * nb1), nb1 / nb0);
        }
        iir1 = iir1_చివర;
    }
}

మూస <రకం_పేరు B, పూర్ణాంకం SUBBLK, తార్కిక HAS_MIN, పూర్ణాంకం BIAS, తార్కిక ACTBIAS>
స్థిర శూన్య ggml_compute_forward_mul_mat_tiled_driver(
        స్థిరరాశి నిర్మాణం ggml_compute_params * params,
              నిర్మాణం ggml_tensor * dst) {

    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    గణన ggml_type      స్థిరరాశి vec_dot_type = ggml_get_type_traits_cpu(src0->type)->vec_dot_type;
    ggml_from_float_t   స్థిరరాశి from_float   = ggml_get_type_traits_cpu(vec_dot_type)->from_float;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb10 == ggml_type_కొలత(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == పరిమాణం(దశాంశం));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    ఒకవేళ (src1->type != vec_dot_type) {
        అక్షరం * wdata = (అక్షరం *) params->wdata;

        స్థిరరాశి size_t nbw0 = ggml_type_కొలత(vec_dot_type);
        స్థిరరాశి size_t nbw1 = ggml_అడ్డువరుస_కొలత(vec_dot_type, ne10);
        స్థిరరాశి size_t nbw2 = nbw1*ne11;
        స్థిరరాశి size_t nbw3 = nbw2*ne12;

        ప్రతిజ్ఞా(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        కొరకు (int64_t i13 = 0; i13 < ne13; ++i13) {
            కొరకు (int64_t i12 = 0; i12 < ne12; ++i12) {
                కొరకు (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_కొలత(vec_dot_type);
                    int64_t ne10_ఖండం_ప్రారంభం = (ith * ne10/bs) / nth;
                    int64_t ne10_ఖండం_చివర   = ((ith + 1) * ne10/bs) / nth;
                    from_float((దశాంశం *)((అక్షరం *) src1->దత్తాంశం + i13*src1->nb[3] + i12*src1->nb[2] + i11*src1->nb[1] + ne10_ఖండం_ప్రారంభం*bs*src1->nb[0]),
                               (శూన్య *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_ఖండం_ప్రారంభం*nbw0),
                               (ne10_ఖండం_చివర - ne10_ఖండం_ప్రారంభం) * bs);
                }
            }
        }
    }

    ఒకవేళ (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth. This saves a bit of coordination right at the start.
        ggml_threadpool_chunk_సమితి(params->threadpool, nth);
    }

    ggml_barrier(params->threadpool);

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    స్థిరరాశి int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    స్థిరరాశి int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    పూర్ణాంకం chunk_కొలత = 256; //TILED_TILE_ROWS;

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim. CEIL(nr/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_కొలత - 1) / chunk_కొలత;
    int64_t nchunk1 = (nr1 + chunk_కొలత - 1) / chunk_కొలత;

    // Step down chunk size if too few chunks to saturate cores, minimum is microtile size
    అయితే (nchunk0 * nchunk1 < nth * 4 && chunk_కొలత > 16) {
        chunk_కొలత = chunk_కొలత / 2;
        nchunk0 = (nr0 + chunk_కొలత - 1) / chunk_కొలత;
        nchunk1 = (nr1 + chunk_కొలత - 1) / chunk_కొలత;
    }

    // The number of elements in each chunk
    స్థిరరాశి int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    స్థిరరాశి int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    పూర్ణాంకం current_chunk = ith;

    // per-thread workspace: after any converted src1 data in wdata, aligned to 64
    అక్షరం * ws_పునాది = (అక్షరం *) params->wdata;
    ఒకవేళ (src1->type != vec_dot_type) {
        ws_పునాది += GGML_PAD(ggml_అడ్డువరుస_కొలత(vec_dot_type, ggml_nelements(src1)), 64);
    }
    ws_పునాది = (అక్షరం *) (((uintptr_t) ws_పునాది + 63) & ~(uintptr_t) 63);
    tiled_ws * ws = (tiled_ws *) (ws_పునాది + (size_t) ith * ggml_tiled_ws_కొలత());

    // TODO:  if we KNOW we're on a machine where all cores are equal, we could skip the coordination/work-stealing and just assign chunks deterministically
    అయితే (current_chunk < nchunk0 * nchunk1) {
        స్థిరరాశి int64_t ith0 = current_chunk % nchunk0;
        స్థిరరాశి int64_t ith1 = current_chunk / nchunk0;

        స్థిరరాశి int64_t ir0_ప్రారంభం = dr0 * ith0;
        స్థిరరాశి int64_t ir0_చివర = MIN(ir0_ప్రారంభం + dr0, nr0);

        స్థిరరాశి int64_t ir1_ప్రారంభం = dr1 * ith1;
        స్థిరరాశి int64_t ir1_చివర = MIN(ir1_ప్రారంభం + dr1, nr1);

        ggml_compute_forward_mul_mat_tiled_one_chunk<B, SUBBLK, HAS_MIN, BIAS, ACTBIAS>(params, dst, ir0_ప్రారంభం, ir0_చివర, ir1_ప్రారంభం, ir1_చివర, ws);

        ఒకవేళ (nth >= nchunk0 * nchunk1) {
            ఆపు;
        }

        current_chunk = ggml_threadpool_chunk_కూడు(params->threadpool, 1);
    }
}

// src0 type dispatch, shared by the MUL_MAT and MUL_MAT_ID entries: one expert for
// MUL_MAT_ID (expert_rows != NULL), the full op for MUL_MAT
మూస <రకం_పేరు B, పూర్ణాంకం SUBBLK, తార్కిక HAS_MIN, పూర్ణాంకం BIAS, తార్కిక ACTBIAS>
స్థిర తార్కిక tiled_matmul_dispatch(స్థిరరాశి నిర్మాణం ggml_compute_params * params,
                                  నిర్మాణం ggml_tensor * dst,
                                  స్థిరరాశి int32_t * expert_అడ్డువరుసలు,
                                  int64_t cur_a,
                                  int64_t cne1,
                                  అక్షరం * scratch) {
    ఒకవేళ (expert_అడ్డువరుసలు == ఖాళీ) {
        ggml_compute_forward_mul_mat_tiled_driver<B, SUBBLK, HAS_MIN, BIAS, ACTBIAS>(params, dst);
    } లేకపోతే {
        ggml_compute_forward_mul_mat_id_tiled_one_expert<B, SUBBLK, HAS_MIN, BIAS, ACTBIAS>(params, dst, cur_a, cne1, expert_అడ్డువరుసలు, scratch);
    }
    తిరిగి నిజం;
}

// the supported src0 types, one list for both ops; false to fall through to vec_dot
స్థిర తార్కిక ggml_tiled_matmul_type_dispatch(స్థిరరాశి నిర్మాణం ggml_compute_params * params,
                                            నిర్మాణం ggml_tensor * dst,
                                            స్థిరరాశి int32_t * expert_అడ్డువరుసలు = ఖాళీ,
                                            int64_t cur_a = 0,
                                            int64_t cne1 = 0,
                                            అక్షరం * scratch = ఖాళీ) {
    ఎంపిక (dst->src[0]->type) {
        స్థితి GGML_TYPE_Q6_K:
            తిరిగి tiled_matmul_dispatch<ఖండం_q6_K, 16, అబద్ధం, 32, నిజం>(params, dst, expert_అడ్డువరుసలు, cur_a, cne1, scratch);
        స్థితి GGML_TYPE_Q5_K:
            తిరిగి tiled_matmul_dispatch<ఖండం_q5_K, 32, నిజం,  0, అబద్ధం>(params, dst, expert_అడ్డువరుసలు, cur_a, cne1, scratch);
        స్థితి GGML_TYPE_Q4_K:
            తిరిగి tiled_matmul_dispatch<ఖండం_q4_K, 32, నిజం,  0, అబద్ధం>(params, dst, expert_అడ్డువరుసలు, cur_a, cne1, scratch);
        స్థితి GGML_TYPE_Q3_K:
            తిరిగి tiled_matmul_dispatch<ఖండం_q3_K, 16, అబద్ధం,  4, నిజం>(params, dst, expert_అడ్డువరుసలు, cur_a, cne1, scratch);
        స్థితి GGML_TYPE_Q2_K:
            తిరిగి tiled_matmul_dispatch<ఖండం_q2_K, 16, నిజం,  0, అబద్ధం>(params, dst, expert_అడ్డువరుసలు, cur_a, cne1, scratch);
        స్థితి GGML_TYPE_IQ4_XS:
            తిరిగి tiled_matmul_dispatch<ఖండం_iq4_xs, 32, అబద్ధం, 128, అబద్ధం>(params, dst, expert_అడ్డువరుసలు, cur_a, cne1, scratch);
        స్థితి GGML_TYPE_IQ2_XXS:
            తిరిగి tiled_matmul_dispatch<ఖండం_iq2_xxs, 32, అబద్ధం, 128, నిజం>(params, dst, expert_అడ్డువరుసలు, cur_a, cne1, scratch);
        స్థితి GGML_TYPE_IQ2_XS:
            తిరిగి tiled_matmul_dispatch<ఖండం_iq2_xs, 16, అబద్ధం, 128, నిజం>(params, dst, expert_అడ్డువరుసలు, cur_a, cne1, scratch);
        స్థితి GGML_TYPE_IQ2_S:
            తిరిగి tiled_matmul_dispatch<ఖండం_iq2_s, 16, అబద్ధం, 128, నిజం>(params, dst, expert_అడ్డువరుసలు, cur_a, cne1, scratch);
        స్థితి GGML_TYPE_IQ3_XXS:
            తిరిగి tiled_matmul_dispatch<ఖండం_iq3_xxs, 32, అబద్ధం, 128, నిజం>(params, dst, expert_అడ్డువరుసలు, cur_a, cne1, scratch);
        స్థితి GGML_TYPE_IQ3_S:
            తిరిగి tiled_matmul_dispatch<ఖండం_iq3_s, 32, అబద్ధం, 128, నిజం>(params, dst, expert_అడ్డువరుసలు, cur_a, cne1, scratch);
        స్థితి GGML_TYPE_IQ1_S:
            తిరిగి tiled_matmul_dispatch<ఖండం_iq1_s, 32, అబద్ధం, 128, నిజం>(params, dst, expert_అడ్డువరుసలు, cur_a, cne1, scratch);
        స్థితి GGML_TYPE_IQ1_M:
            తిరిగి tiled_matmul_dispatch<ఖండం_iq1_m, 16, అబద్ధం, 128, నిజం>(params, dst, expert_అడ్డువరుసలు, cur_a, cne1, scratch);
        అప్రమేయ:
            తిరిగి అబద్ధం;
    }
}

స్థిర తార్కిక ggml_tiled_min_batch(int64_t అడ్డువరుసలు) {
    //  Profitable at rows >= 8, take even when unprofitable if we're forced
    తిరిగి అడ్డువరుసలు >= 8 || ggml_tiled_matmul_forced();
}

// tiled K-quant matmul; returns true if the op was computed here,
// false to fall through to the stock path
తార్కిక ggml_compute_forward_mul_mat_tiled(
        స్థిరరాశి నిర్మాణం ggml_compute_params * params,
              నిర్మాణం ggml_tensor * dst) {
    // --use-ref means bail out and go back to vec_dot reference impl
    ఒకవేళ (params->use_ref) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!ggml_tiled_supported(dst->src[0], dst->src[1])) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!ggml_tiled_min_batch(dst->src[1]->ne[1])) {
        తిరిగి అబద్ధం;
    }
    తిరిగి ggml_tiled_matmul_type_dispatch(params, dst);
}

// MUL_MAT_ID (MoE), one expert; returns true if the expert was computed here,
// per expert eligibility (type gate, batch floor) is decided here
తార్కిక ggml_compute_forward_mul_mat_id_tiled(
        స్థిరరాశి నిర్మాణం ggml_compute_params * params,
              నిర్మాణం ggml_tensor *         dst,
        int64_t                            cur_a,
        int64_t                            cne1,
        స్థిరరాశి int32_t *                    expert_అడ్డువరుసలు,
        అక్షరం *                             scratch) {
    ఒకవేళ (params->use_ref) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!ggml_tiled_supported(dst->src[0], dst->src[1])) {
        తిరిగి అబద్ధం;
    }
    // profitability is per expert: the rows routed to this expert
    ఒకవేళ (!ggml_tiled_min_batch(cne1)) {
        తిరిగి అబద్ధం;
    }
    తిరిగి ggml_tiled_matmul_type_dispatch(params, dst, expert_అడ్డువరుసలు, cur_a, cne1, scratch);
}
