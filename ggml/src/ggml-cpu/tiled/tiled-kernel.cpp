// mulmat microtile kernels

#ಅನ್ತರ್ಭಾವ "tiled-kernel.h"

#ಅನ್ತರ್ಭಾವ "ggml.h"

#ಅನ್ತರ್ಭಾವ <cstring>

#ಆದರೆ defined(__AVX512VNNI__) || defined(__AVX2__) || defined(__AVX__)
#ಅನ್ತರ್ಭಾವ <immintrin.h>
#ಆದರೆ_ಅಂತ್ಯ


// Reference implementation, slower than existing vec_dot approach
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ SUBBLK, ತಾರ್ಕಿಕ HAS_MIN, ಪೂರ್ಣಾಂಕ BIAS, ಪೂರ್ಣಾಂಕ NK>
ಸ್ಥಿರ ಶೂನ್ಯ tiled_ಸತತ_microtile_scalar(ಸ್ಥಿರಾಂಕ tiled_tile_src0 & src0, ಸ್ಥಿರಾಂಕ tiled_tile_src1 & src1,
                                       ಪೂರ್ಣಾಂಕ i0, ಪೂರ್ಣಾಂಕ j0, ಪೂರ್ಣಾಂಕ ಅಂಕೆ_k, ಪೂರ್ಣಾಂಕ slab, ದಶಮಾಂಶ * buf, ಪೂರ್ಣಾಂಕ buf_stride) {
    constexpr ಪೂರ್ಣಾಂಕ NB = TILED_TILE_K / SUBBLK;    // subblocks per 256-K block
    constexpr ಪೂರ್ಣಾಂಕ NS = SUBBLK / 16; // per-16 bsums per subblock

    // num_k/slab: the narrow path holds num_k slabs at row stride num_k*256 (num_k=1, slab=0 = standard)
    // NK: 1 = standard single-slab (offsets 0, strides compile-time); 0 = runtime
    // (num_k/slab from the args). The narrow path is memory-bound, so one runtime
    // version serves all of it.
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nkr  = (NK > 0) ? NK : ಅಂಕೆ_k;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ seff = (NK > 0) ? 0 : slab;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qk_stride = nkr * TILED_TILE_K;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qk_off = seff * TILED_TILE_K;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb_stride = NB * nkr;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb_off = seff * NB;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bs_stride = (nkr == 1) ? TILED_TILE_ಅಡ್ಡಸಾಲುಗಳು : TILED_MICRO;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bs_off = seff * TILED_MICRO;

    ದಶಮಾಂಶ acc[TILED_MICRO][TILED_MICRO];
    memset(acc, 0, ಗಾತ್ರ(acc));

    // subdots at subblock granularity over the 256-K block, exact integer math
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 0; s < NB; s++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < TILED_MICRO; j++) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ br = j0 + j;
            ಸ್ಥಿರಾಂಕ int8_t * q1 = &src1.q[br * qk_stride + qk_off + s * SUBBLK];
            int32_t bsum = 0;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ u = 0; u < NS; u++) {
                bsum += src1.bsums[(bs_off + s * NS + u) * bs_stride + br];
            }

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < TILED_MICRO; i++) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ar = i0 + i;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ d_off = seff * TILED_MICRO + ar;
                ಸ್ಥಿರಾಂಕ uint8_t * q0 = &src0.q[ar * qk_stride + qk_off + s * SUBBLK];

                int32_t raw = 0;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ e = 0; e < SUBBLK; e++) {
                    raw += (int32_t) q0[e] * (int32_t) q1[e];
                }

                // BIAS: subtract BIAS*bsum (src1's per-subblock code sum) from the exact int raw
                int32_t corr = raw;
                ಆದರೆ constexpr (BIAS != 0) {
                    corr -= BIAS * bsum;
                }
                ಸ್ಥಿರಾಂಕ int32_t scales_raw = (int32_t) src0.scales[ar * nb_stride + nb_off + s] * corr;
                // d is NOT applied here: it is constant over the s-loop, so we apply it last before write-out
                ಆದರೆ constexpr (HAS_MIN) {
                    ಸ್ಥಿರಾಂಕ int32_t mins_bsum = (int32_t) src0.mins[ar * nb_stride + nb_off + s] * bsum;
                    acc[i][j] += (ದಶಮಾಂಶ) src0.d[d_off] * (ದಶಮಾಂಶ) scales_raw
                              - (ದಶಮಾಂಶ) src0.dmin[d_off] * (ದಶಮಾಂಶ) mins_bsum;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    acc[i][j] += (ದಶಮಾಂಶ) src0.d[d_off] * (ದಶಮಾಂಶ) scales_raw;
                }
            }
        }
    }

    // Apply d and write out to buf
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < TILED_MICRO; i++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < TILED_MICRO; j++) {
            buf[(i0 + i) * buf_stride + (j0 + j)] += src1.d[seff * TILED_MICRO + j0 + j] * acc[i][j];
        }
    }
}

#ಆದರೆ defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)


// VNNI microkernel
// 8x16 band pass: 8 src0 rows (i0, i0+8) x 16 src1 cols (j0, j0+16)
// Math: s1_acc += scales_s*raw_s, s2_acc += mins_s*bsums_s per subblock; the row result
// is d0*s1_acc - dmin*s2_acc (f32, exact in range), applied once per row.
//
// Register pressure: the band pass holds acc16 + s1_acc = 16 zmm for the
// 8-row band (s2_acc has no dpbusd dependency, so it is computed after the
// band pass and adds no register pressure);
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ SUBBLK, ತಾರ್ಕಿಕ HAS_MIN, ಪೂರ್ಣಾಂಕ BIAS, ಪೂರ್ಣಾಂಕ NK>
ಸ್ಥಿರ ಶೂನ್ಯ tiled_ಸತತ_micro_vnni_8x16(ಸ್ಥಿರಾಂಕ tiled_tile_src0 & src0, ಸ್ಥಿರಾಂಕ tiled_tile_src1 & src1,
                                          ಪೂರ್ಣಾಂಕ i0, ಪೂರ್ಣಾಂಕ j0, ಪೂರ್ಣಾಂಕ ಅಂಕೆ_k, ಪೂರ್ಣಾಂಕ slab, ದಶಮಾಂಶ * buf, ಪೂರ್ಣಾಂಕ buf_stride) {
    constexpr ಪೂರ್ಣಾಂಕ NB = TILED_TILE_K / SUBBLK;
    constexpr ಪೂರ್ಣಾಂಕ NS = SUBBLK / 16;
    constexpr ಪೂರ್ಣಾಂಕ NG = SUBBLK / 4;

    // band width, see the register-pressure note above
    constexpr ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು = 8;

    // num_k = K-blocks per row (the narrow path holds num_k slabs at row stride num_k*256 so
    // each weight row is one long stream); slab = this call's slab. num_k=1, slab=0 = standard.
    // NK: 1 = standard single-slab (offsets 0, strides compile-time); 0 = runtime
    // (num_k/slab from the args). The narrow path is memory-bound, so one runtime
    // version serves all of it.
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nkr  = (NK > 0) ? NK : ಅಂಕೆ_k;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ seff = (NK > 0) ? 0 : slab;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qk_stride = nkr * TILED_TILE_K;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qk_off = seff * TILED_TILE_K;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb_stride = NB * nkr;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb_off = seff * NB;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bs_stride = (nkr == 1) ? TILED_TILE_ಅಡ್ಡಸಾಲುಗಳು : TILED_MICRO;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bs_off = seff * TILED_MICRO;

    ಸ್ಥಿರಾಂಕ __m512 d1_vec = mm512_ತುಂಬು_ps(&src1.d[seff * TILED_MICRO + j0]);

    __m512i s1_acc[ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು];
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು; t++) { s1_acc[t] = _mm512_setzero_si512(); }
    __m512i s2_acc[ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು];
    ಆದರೆ constexpr (HAS_MIN) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು; t++) { s2_acc[t] = _mm512_setzero_si512(); }
    }

    ಸ್ಥಿರಾಂಕ int32_t * bsums = src1.bsums; // int32 per-16 sums; NS > 1 combines NS lanes per subblock
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 0; s < NB; s++) {
        __m512i bsums32 = mm512_ತುಂಬು_si512((ಸ್ಥಿರಾಂಕ __m512i *) &bsums[(bs_off + s * NS) * bs_stride + j0]);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ u = 1; u < NS; u++) {
            bsums32 = mm512_ಕೂಡಿಸು_epi32(bsums32, mm512_ತುಂಬು_si512((ಸ್ಥಿರಾಂಕ __m512i *) &bsums[(bs_off + s * NS + u) * bs_stride + j0]));
        }

        __m512i bias32 = _mm512_setzero_si512();
        ಆದರೆ constexpr (BIAS != 0) {
            bias32 = _mm512_mullo_epi32(bsums32, _mm512_set1_epi32(BIAS));
        }

        __m512i acc16[ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು];
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು; t++) { acc16[t] = _mm512_setzero_si512(); }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __GNUC__
        #ಸೂಚನೆ GCC unroll 8 // pragma unrolled justified by measuing with/without
#ಆದರೆ_ಅಂತ್ಯ
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ g = 0; g < NG; g++) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kg = s * NG + g;
            // in-place interleave layout: [kg%16 @ TILED_TILE_K][kg/16 @ 64][row @ 4]
            ಸ್ಥಿರಾಂಕ __m512i codes = mm512_ತುಂಬು_si512((ಸ್ಥಿರಾಂಕ __m512i *) &src1.q[(j0 / TILED_MICRO) * (TILED_MICRO * qk_stride)
                + qk_off + (kg % TILED_MICRO) * qk_stride + (kg / TILED_MICRO) * (TILED_MICRO * 4)]);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು; t++) {
                ಸ್ಥಿರಾಂಕ uint32_t u4 = *(ಸ್ಥಿರಾಂಕ uint32_t *) &src0.q[(i0 + t) * qk_stride + qk_off + kg * 4];
                ಸ್ಥಿರಾಂಕ __m512i u4b = _mm512_set1_epi32((ಪೂರ್ಣಾಂಕ) u4);
                acc16[t] = _mm512_dpbusd_epi32(acc16[t], u4b, codes);
            }
        }

        // int correction: s1_acc += scales*(raw-BIAS*bsums)
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು; t++) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ar = i0 + t;
            __m512i rawi = acc16[t];
            ಆದರೆ constexpr (BIAS != 0) {
                rawi = _mm512_sub_epi32(rawi, bias32);
            }
            s1_acc[t] = mm512_ಕೂಡಿಸು_epi32(s1_acc[t],
                _mm512_mullo_epi32(rawi, _mm512_set1_epi32(src0.scales[ar * nb_stride + nb_off + s])));
        }
    }

    // s2_acc += mins*bsums; independent of the dpbusd results, so it runs here instead of in the
    // band pass: the band pass keeps the register budget for the 8-row band
    ಆದರೆ constexpr (HAS_MIN) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 0; s < NB; s++) {
            __m512i bsums32 = mm512_ತುಂಬು_si512((ಸ್ಥಿರಾಂಕ __m512i *) &bsums[(bs_off + s * NS) * bs_stride + j0]);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ u = 1; u < NS; u++) {
                bsums32 = mm512_ಕೂಡಿಸು_epi32(bsums32, mm512_ತುಂಬು_si512((ಸ್ಥಿರಾಂಕ __m512i *) &bsums[(bs_off + s * NS + u) * bs_stride + j0]));
            }
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು; t++) {
                s2_acc[t] = mm512_ಕೂಡಿಸು_epi32(s2_acc[t],
                    _mm512_mullo_epi32(bsums32, _mm512_set1_epi32(src0.mins[(i0 + t) * nb_stride + nb_off + s])));
            }
        }
    }

    // epilogue: int->float, apply per-row scales, store to buf
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು; t++) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ar = i0 + t;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ d_off = seff * TILED_MICRO + ar;
        __m512 f1 = _mm512_cvtepi32_ps(s1_acc[t]);
        __m512 ಫಲಿತಾಂಶ = _mm512_mul_ps(f1, _mm512_set1_ps(src0.d[d_off]));
        ಆದರೆ constexpr (HAS_MIN) {
            __m512 f2 = _mm512_cvtepi32_ps(s2_acc[t]);
            ಫಲಿತಾಂಶ = _mm512_fnmadd_ps(_mm512_set1_ps(src0.dmin[d_off]), f2, ಫಲಿತಾಂಶ);
        }
        ದಶಮಾಂಶ * p = &buf[(i0 + t) * buf_stride + j0];
        _mm512_store_ps(p, mm512_ಕೂಡಿಸು_ps(mm512_ತುಂಬು_ps(p), _mm512_mul_ps(ಫಲಿತಾಂಶ, d1_vec)));
    }
}

// 16x16 microtile as two explicit 8x16 band passes
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ SUBBLK, ತಾರ್ಕಿಕ HAS_MIN, ಪೂರ್ಣಾಂಕ BIAS, ಪೂರ್ಣಾಂಕ NK>
ಸ್ಥಿರ ಶೂನ್ಯ tiled_ಸತತ_microtile_vnni(ಸ್ಥಿರಾಂಕ tiled_tile_src0 & src0, ಸ್ಥಿರಾಂಕ tiled_tile_src1 & src1,
                                         ಪೂರ್ಣಾಂಕ i0, ಪೂರ್ಣಾಂಕ j0, ಪೂರ್ಣಾಂಕ ಅಂಕೆ_k, ಪೂರ್ಣಾಂಕ slab, ದಶಮಾಂಶ * buf, ಪೂರ್ಣಾಂಕ buf_stride) {
    tiled_ಸತತ_micro_vnni_8x16<SUBBLK, HAS_MIN, BIAS, NK>(src0, src1, i0,      j0, ಅಂಕೆ_k, slab, buf, buf_stride);
    tiled_ಸತತ_micro_vnni_8x16<SUBBLK, HAS_MIN, BIAS, NK>(src0, src1, i0 + 8,  j0, ಅಂಕೆ_k, slab, buf, buf_stride);
}

#ಆದರೆ_ಅಂತ್ಯ // __AVX512VNNI__ && __AVX512VL__

#ಆದರೆ defined(__AVX2__)

// AVX2 kernel.
// We're effectively applying the existing vec_dot algorithms to an 8x16 block here.
// Different paths based on subblock size as it affects when/where we multiply in scales and apply mins
// SUBBLK=32: one 256-bit maddubs per (s, column), a 256-bit set1 scale,
// one 8-lane i32 accumulator per column.
// SUBBLK=16: two subblocks per 32B load. The 256-bit maddubs product
// still gives 16 i16 lanes (0..7 = subblock sp, 8..15 = sp+1), but the
// scale is applied per 128-bit half.
// With BIAS != 0 the q0 codes are biased (up to 241) and a maddubs i16 pair
// overflows, so the weight is debiased (w = q0 - BIAS, small). Two MACs:
//   ACTBIAS: activation pre-biased (+128) by repack, one maddubs(q1b, w); the +128 is
//     corrected out in the epilogue (128*sum_s scales[s]*w_bsum[s], precomputed in repack_src0).
//     Fits only when the debiased weight magnitude Wmax <= 64 (2*255*Wmax < 32767).
//   else (iq4_xs, Wmax = 127): the signed-int8 sign trick, ax = |w|, sy = q1*sign(w),
//     one maddubs (2*127*127 < 32767) so it always fits.
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ SUBBLK, ತಾರ್ಕಿಕ HAS_MIN, ಪೂರ್ಣಾಂಕ BIAS, ಪೂರ್ಣಾಂಕ NK, ತಾರ್ಕಿಕ ACTBIAS>
ಸ್ಥಿರ ಶೂನ್ಯ tiled_ಸತತ_microtile_avx2(ಸ್ಥಿರಾಂಕ tiled_tile_src0 & src0, ಸ್ಥಿರಾಂಕ tiled_tile_src1 & src1,
                                     ಪೂರ್ಣಾಂಕ i0, ಪೂರ್ಣಾಂಕ j0, ಪೂರ್ಣಾಂಕ ಅಂಕೆ_k, ಪೂರ್ಣಾಂಕ slab, ದಶಮಾಂಶ * buf, ಪೂರ್ಣಾಂಕ buf_stride) {
    constexpr ಪೂರ್ಣಾಂಕ NB = TILED_TILE_K / SUBBLK;
    constexpr ಪೂರ್ಣಾಂಕ ಗುಂಪು_2 = 8; // Process 8 src1 columns concurrently (1 YMM vector)

    static_assert(SUBBLK == 16 || SUBBLK == 32, "unsupported SUBBLK");

    // num_k/slab: the narrow path holds num_k slabs at row stride num_k*256 (num_k=1, slab=0 = standard)
    // NK: 1 = standard single-slab (offsets 0, strides compile-time); 0 = runtime
    // (num_k/slab from the args). The narrow path is memory-bound, so one runtime
    // version serves all of it.
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nkr  = (NK > 0) ? NK : ಅಂಕೆ_k;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ seff = (NK > 0) ? 0 : slab;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qk_stride = nkr * TILED_TILE_K;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qk_off = seff * TILED_TILE_K;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb_stride = NB * nkr;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb_off = seff * NB;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bs_stride = (nkr == 1) ? TILED_TILE_ಅಡ್ಡಸಾಲುಗಳು : TILED_MICRO;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bs_off = seff * TILED_MICRO;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < TILED_MICRO; i++) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ar = i0 + i;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ d_off = seff * TILED_MICRO + ar;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d0 = src0.d[d_off];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin0 = src0.dmin[d_off];
        ಸ್ಥಿರಾಂಕ uint8_t * q0 = &src0.q[ar * qk_stride + qk_off];
        ಸ್ಥಿರಾಂಕ int32_t * scales_ಅಡ್ಡಸಾಲು = &src0.scales[ar * nb_stride + nb_off];
        ಸ್ಥಿರಾಂಕ int32_t * mins_ಅಡ್ಡಸಾಲು = &src0.mins[ar * nb_stride + nb_off];

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ g = 0; g < TILED_MICRO; g += ಗುಂಪು_2) {
            // Pre-calculate weight pointers for this column group
            ಸ್ಥಿರಾಂಕ int8_t * q1_ptr[ಗುಂಪು_2];
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < ಗುಂಪು_2; t++) {
                q1_ptr[t] = &src1.q[(j0 + g + t) * qk_stride + qk_off];
            }

            __m256i s2   = _mm256_setzero_si256();

            __m256i acc[ಗುಂಪು_2];
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < ಗುಂಪು_2; t++) {
                acc[t] = _mm256_setzero_si256();
            }

            ಆದರೆ constexpr (SUBBLK == 32) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 0; s < NB; s++) {
                    // OPTIMIZATION 1: Use aligned loads (_mm256_load_si256)
                    ಸ್ಥಿರಾಂಕ __m256i q0_32 = mm256_ತುಂಬು_si256((ಸ್ಥಿರಾಂಕ __m256i *) &q0[s * SUBBLK]);
                    ಸ್ಥಿರಾಂಕ __m256i scales16 = _mm256_set1_epi16(scales_ಅಡ್ಡಸಾಲು[s]);

                    ಆದರೆ constexpr (BIAS != 0) {
                        ಸ್ಥಿರಾಂಕ __m256i w = _mm256_sub_epi8(q0_32, _mm256_set1_epi8((int8_t) BIAS)); // debiased weight
                        ಆದರೆ constexpr (ACTBIAS) {
                            // activation pre-biased (+128) by repack; q1b unsigned, w signed
                            #ಸೂಚನೆ GCC unroll 8
                            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < ಗುಂಪು_2; t++) {
                                ಸ್ಥಿರಾಂಕ __m256i q1b_32 = mm256_ತುಂಬು_si256((ಸ್ಥಿರಾಂಕ __m256i *) &q1_ptr[t][s * SUBBLK]);
                                acc[t] = mm256_ಕೂಡಿಸು_epi32(acc[t],
                                    _mm256_madd_epi16(scales16, _mm256_maddubs_epi16(q1b_32, w)));
                            }
                        } ಇಲ್ಲದಿದ್ದರೆ {
                            // iq4_xs (Wmax > 64 overflows act-bias): sign trick, both operands <= 127
                            ಸ್ಥಿರಾಂಕ __m256i ax = _mm256_sign_epi8(w, w);
                            #ಸೂಚನೆ GCC unroll 8
                            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < ಗುಂಪು_2; t++) {
                                ಸ್ಥಿರಾಂಕ __m256i q1_32 = mm256_ತುಂಬು_si256((ಸ್ಥಿರಾಂಕ __m256i *) &q1_ptr[t][s * SUBBLK]);
                                acc[t] = mm256_ಕೂಡಿಸು_epi32(acc[t],
                                    _mm256_madd_epi16(scales16, _mm256_maddubs_epi16(ax, _mm256_sign_epi8(q1_32, w))));
                            }
                        }
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        #ಸೂಚನೆ GCC unroll 8
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < ಗುಂಪು_2; t++) {
                            ಸ್ಥಿರಾಂಕ __m256i q1_32 = mm256_ತುಂಬು_si256((ಸ್ಥಿರಾಂಕ __m256i *) &q1_ptr[t][s * SUBBLK]);
                            acc[t] = mm256_ಕೂಡಿಸು_epi32(acc[t],
                                _mm256_madd_epi16(scales16, _mm256_maddubs_epi16(q0_32, q1_32)));
                        }
                    }

                    ಆದರೆ constexpr (HAS_MIN) {
                        ಸ್ಥಿರಾಂಕ __m256i bsums_v = mm256_ಕೂಡಿಸು_epi32(
                            mm256_ತುಂಬು_si256((ಸ್ಥಿರಾಂಕ __m256i *) &src1.bsums[(bs_off + s * 2) * bs_stride + j0 + g]),
                            mm256_ತುಂಬು_si256((ಸ್ಥಿರಾಂಕ __m256i *) &src1.bsums[(bs_off + s * 2 + 1) * bs_stride + j0 + g]));
                        s2 = mm256_ಕೂಡಿಸು_epi32(s2, _mm256_mullo_epi32(bsums_v, _mm256_set1_epi32(mins_ಅಡ್ಡಸಾಲು[s])));
                    }
                }
            } ಇಲ್ಲದಿದ್ದರೆ { // SUBBLK == 16
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ sp = 0; sp < NB; sp += 2) {
                    ಸ್ಥಿರಾಂಕ __m256i q0_32 = mm256_ತುಂಬು_si256((ಸ್ಥಿರಾಂಕ __m256i *) &q0[sp * SUBBLK]);
                    ಸ್ಥಿರಾಂಕ __m256i scalesv = mm256_ಗಣ_m128i(_mm_set1_epi16(scales_ಅಡ್ಡಸಾಲು[sp + 1]), _mm_set1_epi16(scales_ಅಡ್ಡಸಾಲು[sp]));
                    ಆದರೆ constexpr (BIAS != 0) {
                        ಸ್ಥಿರಾಂಕ __m256i w = _mm256_sub_epi8(q0_32, _mm256_set1_epi8((int8_t) BIAS)); // debiased weight
                        ಆದರೆ constexpr (ACTBIAS) {
                            // activation pre-biased (+128) by repack; q1b unsigned, w signed
                            #ಸೂಚನೆ GCC unroll 8
                            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < ಗುಂಪು_2; t++) {
                                ಸ್ಥಿರಾಂಕ __m256i q1b_32 = mm256_ತುಂಬು_si256((ಸ್ಥಿರಾಂಕ __m256i *) &q1_ptr[t][sp * SUBBLK]);
                                acc[t] = mm256_ಕೂಡಿಸು_epi32(acc[t], _mm256_madd_epi16(
                                    scalesv, _mm256_maddubs_epi16(q1b_32, w)));
                            }
                        } ಇಲ್ಲದಿದ್ದರೆ {
                            // iq4_xs (Wmax > 64 overflows act-bias): sign trick, both operands <= 127
                            ಸ್ಥಿರಾಂಕ __m256i ax = _mm256_sign_epi8(w, w);
                            #ಸೂಚನೆ GCC unroll 8
                            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < ಗುಂಪು_2; t++) {
                                ಸ್ಥಿರಾಂಕ __m256i q1_32 = mm256_ತುಂಬು_si256((ಸ್ಥಿರಾಂಕ __m256i *) &q1_ptr[t][sp * SUBBLK]);
                                acc[t] = mm256_ಕೂಡಿಸು_epi32(acc[t], _mm256_madd_epi16(
                                    scalesv, _mm256_maddubs_epi16(ax, _mm256_sign_epi8(q1_32, w))));
                            }
                        }
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        #ಸೂಚನೆ GCC unroll 8
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < ಗುಂಪು_2; t++) {
                            ಸ್ಥಿರಾಂಕ __m256i q1_32 = mm256_ತುಂಬು_si256((ಸ್ಥಿರಾಂಕ __m256i *) &q1_ptr[t][sp * SUBBLK]);
                            acc[t] = mm256_ಕೂಡಿಸು_epi32(acc[t], _mm256_madd_epi16(
                                scalesv, _mm256_maddubs_epi16(q0_32, q1_32)));
                        }
                    }

                    ಆದರೆ constexpr (HAS_MIN) {
                        ಸ್ಥಿರಾಂಕ __m256i bsums0_v = mm256_ತುಂಬು_si256((ಸ್ಥಿರಾಂಕ __m256i *) &src1.bsums[(bs_off + sp) * bs_stride + j0 + g]);
                        ಸ್ಥಿರಾಂಕ __m256i bsums1_v = mm256_ತುಂಬು_si256((ಸ್ಥಿರಾಂಕ __m256i *) &src1.bsums[(bs_off + sp + 1) * bs_stride + j0 + g]);
                        s2 = mm256_ಕೂಡಿಸು_epi32(s2, mm256_ಕೂಡಿಸು_epi32(
                            _mm256_mullo_epi32(bsums0_v, _mm256_set1_epi32(mins_ಅಡ್ಡಸಾಲು[sp])),
                            _mm256_mullo_epi32(bsums1_v, _mm256_set1_epi32(mins_ಅಡ್ಡಸಾಲು[sp + 1]))));
                    }
                }
            }

            // Reduce the eight accumulators to column sums in acc[0]..acc[7] order, with additions modulo 2^32.
            ಸ್ಥಿರಾಂಕ __m256i ab = _mm256_hadd_epi32(acc[0], acc[1]);
            ಸ್ಥಿರಾಂಕ __m256i cd = _mm256_hadd_epi32(acc[2], acc[3]);
            ಸ್ಥಿರಾಂಕ __m256i ef = _mm256_hadd_epi32(acc[4], acc[5]);
            ಸ್ಥಿರಾಂಕ __m256i gh = _mm256_hadd_epi32(acc[6], acc[7]);

            // Each 128-bit half now holds four column sums over input lanes 0..3 or 4..7.
            ಸ್ಥಿರಾಂಕ __m256i abcd = _mm256_hadd_epi32(ab, cd);
            ಸ್ಥಿರಾಂಕ __m256i efgh = _mm256_hadd_epi32(ef, gh);
            ಸ್ಥಿರಾಂಕ __m256i lo = _mm256_permute2x128_si256(abcd, efgh, 0x20);
            ಸ್ಥಿರಾಂಕ __m256i hi = _mm256_permute2x128_si256(abcd, efgh, 0x31);
            __m256i s1_vec = mm256_ಕೂಡಿಸು_epi32(lo, hi);

            // OPTIMIZATION 2: Vectorized Epilogue with FMA3 gating
            ಆದರೆ constexpr (BIAS != 0 && ACTBIAS) {
                // act-bias correction: raw = desired + 128*sum_s scales[s]*w_bsum[s]; corr precomputed in repack_src0
                s1_vec = _mm256_sub_epi32(s1_vec, _mm256_set1_epi32(mins_ಅಡ್ಡಸಾಲು[0]));
            }

            ಸ್ಥಿರಾಂಕ __m256 src1_d_vec = mm256_ತುಂಬು_ps(&src1.d[seff * TILED_MICRO + j0 + g]);
            __m256 res_vec = _mm256_mul_ps(_mm256_set1_ps(d0), _mm256_cvtepi32_ps(s1_vec));

            ಆದರೆ constexpr (HAS_MIN) {
#ಆದರೆ defined(__FMA__)
                // FMA3 implementation: res_vec = res_vec - (dmin0 * s2)
                res_vec = _mm256_fnmadd_ps(_mm256_set1_ps(dmin0), _mm256_cvtepi32_ps(s2), res_vec);
#ಇಲ್ಲದಿದ್ದರೆ
                // Non-FMA fallback
                res_vec = _mm256_sub_ps(res_vec, _mm256_mul_ps(_mm256_set1_ps(dmin0), _mm256_cvtepi32_ps(s2)));
#ಆದರೆ_ಅಂತ್ಯ
            }

            ದಶಮಾಂಶ * buf_ptr = &buf[ar * buf_stride + j0 + g];
            __m256 current_buf = _mm256_loadu_ps(buf_ptr);

#ಆದರೆ defined(__FMA__)
            // FMA3 accumulation: current_buf + (res_vec * src1_d_vec)
            __m256 updated_buf = _mm256_fmadd_ps(res_vec, src1_d_vec, current_buf);
#ಇಲ್ಲದಿದ್ದರೆ
            __m256 updated_buf = mm256_ಕೂಡಿಸು_ps(current_buf, _mm256_mul_ps(res_vec, src1_d_vec));
#ಆದರೆ_ಅಂತ್ಯ

            _mm256_storeu_ps(buf_ptr, updated_buf);
        }
    }
}

#ಆದರೆ_ಅಂತ್ಯ // __AVX2__

#ಆದರೆ defined(__AVX__) && !defined(__AVX2__)
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ SUBBLK, ತಾರ್ಕಿಕ HAS_MIN, ಪೂರ್ಣಾಂಕ BIAS, ಪೂರ್ಣಾಂಕ NK>
ಸ್ಥಿರ ಶೂನ್ಯ tiled_ಸತತ_microtile_avx(ಸ್ಥಿರಾಂಕ tiled_tile_src0 & src0, ಸ್ಥಿರಾಂಕ tiled_tile_src1 & src1,
                                    ಪೂರ್ಣಾಂಕ i0, ಪೂರ್ಣಾಂಕ j0, ಪೂರ್ಣಾಂಕ ಅಂಕೆ_k, ಪೂರ್ಣಾಂಕ slab, ದಶಮಾಂಶ * buf, ಪೂರ್ಣಾಂಕ buf_stride) {
    constexpr ಪೂರ್ಣಾಂಕ NB = TILED_TILE_K / SUBBLK;
    constexpr ಪೂರ್ಣಾಂಕ NS = SUBBLK / 16;
    constexpr ಪೂರ್ಣಾಂಕ ಗುಂಪು_2 = 4; // src1 columns per group: one acc32 per column

    // num_k/slab: the narrow path holds num_k slabs at row stride num_k*256 (num_k=1, slab=0 = standard)
    // NK: 1 = standard single-slab (offsets 0, strides compile-time); 0 = runtime
    // (num_k/slab from the args). The narrow path is memory-bound, so one runtime
    // version serves all of it.
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nkr  = (NK > 0) ? NK : ಅಂಕೆ_k;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ seff = (NK > 0) ? 0 : slab;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qk_stride = nkr * TILED_TILE_K;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qk_off = seff * TILED_TILE_K;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb_stride = NB * nkr;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb_off = seff * NB;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bs_stride = (nkr == 1) ? TILED_TILE_ಅಡ್ಡಸಾಲುಗಳು : TILED_MICRO;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bs_off = seff * TILED_MICRO;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < TILED_MICRO; i++) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ar = i0 + i;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ d_off = seff * TILED_MICRO + ar;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d0 = src0.d[d_off];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin0 = src0.dmin[d_off];
        ಸ್ಥಿರಾಂಕ uint8_t * q0 = &src0.q[ar * qk_stride + qk_off];
        ಸ್ಥಿರಾಂಕ int32_t * scales_ಅಡ್ಡಸಾಲು = &src0.scales[ar * nb_stride + nb_off];
        ಸ್ಥಿರಾಂಕ int32_t * mins_ಅಡ್ಡಸಾಲು = &src0.mins[ar * nb_stride + nb_off];

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ g = 0; g < TILED_MICRO; g += ಗುಂಪು_2) {
            ಸ್ಥಿರಾಂಕ int8_t * q1g[ಗುಂಪು_2];
            __m128i acc[ಗುಂಪು_2];
            __m128i s2 = _mm_setzero_si128();   // sum_s mins_s * bsums_s
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < ಗುಂಪು_2; t++) {
                q1g[t] = &src1.q[(j0 + g + t) * qk_stride + qk_off];
                acc[t] = _mm_setzero_si128();
            }

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 0; s < NB; s++) {
                ಸ್ಥಿರಾಂಕ __m128i scales16 = _mm_set1_epi16(scales_ಅಡ್ಡಸಾಲು[s]);
                ಆದರೆ constexpr (HAS_MIN) {
                    __m128i bsums_v = _mm_setzero_si128(); // per-col bsums as a 4-lane i32 vector
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ u = 0; u < NS; u++) {
                        bsums_v = mm_ಕೂಡಿಸು_epi32(bsums_v, _mm_loadu_si128(
                            (ಸ್ಥಿರಾಂಕ __m128i *) &src1.bsums[(bs_off + s * NS + u) * bs_stride + j0 + g]));
                    }
                    s2 = mm_ಕೂಡಿಸು_epi32(s2, _mm_mullo_epi32(bsums_v, _mm_set1_epi32(mins_ಅಡ್ಡಸಾಲು[s])));
                }
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ u = 0; u < NS; u++) {
                    ಸ್ಥಿರಾಂಕ __m128i a16 = _mm_loadu_si128((ಸ್ಥಿರಾಂಕ __m128i *) &q0[s * SUBBLK + u * 16]);
                    ಆದರೆ constexpr (BIAS != 0) {
                        // debiased weight w = a16 - BIAS; signed dot via the sign trick (see header)
                        ಸ್ಥಿರಾಂಕ __m128i a16s = _mm_sub_epi8(a16, _mm_set1_epi8((int8_t) BIAS));
                        ಸ್ಥಿರಾಂಕ __m128i ax    = _mm_sign_epi8(a16s, a16s);
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < ಗುಂಪು_2; t++) {
                            ಸ್ಥಿರಾಂಕ __m128i q1_16 = _mm_loadu_si128((ಸ್ಥಿರಾಂಕ __m128i *) &q1g[t][s * SUBBLK + u * 16]);
                            acc[t] = mm_ಕೂಡಿಸು_epi32(acc[t],
                                _mm_madd_epi16(scales16, _mm_maddubs_epi16(ax, _mm_sign_epi8(q1_16, a16s))));
                        }
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < ಗುಂಪು_2; t++) {
                            acc[t] = mm_ಕೂಡಿಸು_epi32(acc[t], _mm_madd_epi16(scales16,
                                _mm_maddubs_epi16(a16,
                                    _mm_loadu_si128((ಸ್ಥಿರಾಂಕ __m128i *) &q1g[t][s * SUBBLK + u * 16]))));
                        }
                    }
                }
            }

            int32_t s2_s[ಗುಂಪು_2] = { 0, 0, 0, 0 };
            ಆದರೆ constexpr (HAS_MIN) {
                _mm_storeu_si128((__m128i *) s2_s, s2);
            }

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < ಗುಂಪು_2; t++) {
                // 4 i32 lanes -> scalar
                __m128i v = _mm_shuffle_epi32(acc[t], _MM_SHUFFLE(2, 3, 0, 1));
                acc[t] = mm_ಕೂಡಿಸು_epi32(acc[t], v);
                v = _mm_shuffle_epi32(acc[t], _MM_SHUFFLE(1, 0, 3, 2));
                acc[t] = mm_ಕೂಡಿಸು_epi32(acc[t], v);
                ಸ್ಥಿರಾಂಕ int32_t s1 = _mm_cvtsi128_si32(acc[t]);

                ದಶಮಾಂಶ res = d0 * (ದಶಮಾಂಶ) s1;
                ಆದರೆ constexpr (HAS_MIN) {
                    res -= dmin0 * (ದಶಮಾಂಶ) s2_s[t];
                }
                buf[ar * buf_stride + j0 + g + t] += src1.d[seff * TILED_MICRO + j0 + g + t] * res;
            }
        }
    }
}

#ಆದರೆ_ಅಂತ್ಯ // __AVX__ && !__AVX2__

// main microtile entry point. num_k = K-blocks per row (the narrow path reads num_k slabs at row
// stride num_k*256 so each weight row is one long stream); slab = this call's slab. num_k=1,
// slab=0 is the standard single-slab path.
// num_k==1 dispatches to the NK=1 instantiation, whose strides/offsets are compile-time
// constants (shifts/scaled-leas). num_k>1 (the narrow path) uses the NK=0 runtime-strided
// version; it is memory-bound, so one version serves all of it.
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ SUBBLK, ತಾರ್ಕಿಕ HAS_MIN, ಪೂರ್ಣಾಂಕ BIAS, ತಾರ್ಕಿಕ ACTBIAS>
ಶೂನ್ಯ tiled_ಸತತ_microtile(ಸ್ಥಿರಾಂಕ tiled_tile_src0 & src0, ಸ್ಥಿರಾಂಕ tiled_tile_src1 & src1,
                         ಪೂರ್ಣಾಂಕ i0, ಪೂರ್ಣಾಂಕ j0, ಪೂರ್ಣಾಂಕ ಅಂಕೆ_k, ಪೂರ್ಣಾಂಕ slab, ದಶಮಾಂಶ * buf, ಪೂರ್ಣಾಂಕ buf_stride) {
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ standard = (ಅಂಕೆ_k == 1);
#ಆದರೆ defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)
    ಆದರೆ (standard) {
        tiled_ಸತತ_microtile_vnni<SUBBLK, HAS_MIN, BIAS, 1>(src0, src1, i0, j0, ಅಂಕೆ_k, slab, buf, buf_stride);
    } ಇಲ್ಲದಿದ್ದರೆ {
        tiled_ಸತತ_microtile_vnni<SUBBLK, HAS_MIN, BIAS, 0>(src0, src1, i0, j0, ಅಂಕೆ_k, slab, buf, buf_stride);
    }
#ಅಥವಾಆದರೆ defined(__AVX2__)
    ಆದರೆ (standard) {
        tiled_ಸತತ_microtile_avx2<SUBBLK, HAS_MIN, BIAS, 1, ACTBIAS>(src0, src1, i0, j0, ಅಂಕೆ_k, slab, buf, buf_stride);
    } ಇಲ್ಲದಿದ್ದರೆ {
        tiled_ಸತತ_microtile_avx2<SUBBLK, HAS_MIN, BIAS, 0, ACTBIAS>(src0, src1, i0, j0, ಅಂಕೆ_k, slab, buf, buf_stride);
    }
#ಅಥವಾಆದರೆ defined(__AVX__)
    ಆದರೆ (standard) {
        tiled_ಸತತ_microtile_avx<SUBBLK, HAS_MIN, BIAS, 1>(src0, src1, i0, j0, ಅಂಕೆ_k, slab, buf, buf_stride);
    } ಇಲ್ಲದಿದ್ದರೆ {
        tiled_ಸತತ_microtile_avx<SUBBLK, HAS_MIN, BIAS, 0>(src0, src1, i0, j0, ಅಂಕೆ_k, slab, buf, buf_stride);
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ಆದರೆ (standard) {
        tiled_ಸತತ_microtile_scalar<SUBBLK, HAS_MIN, BIAS, 1>(src0, src1, i0, j0, ಅಂಕೆ_k, slab, buf, buf_stride);
    } ಇಲ್ಲದಿದ್ದರೆ {
        tiled_ಸತತ_microtile_scalar<SUBBLK, HAS_MIN, BIAS, 0>(src0, src1, i0, j0, ಅಂಕೆ_k, slab, buf, buf_stride);
    }
#ಆದರೆ_ಅಂತ್ಯ
}

// explicit instantiations for the in-use formats (q4_K and q5_K share the constants)
// ACTBIAS selects the act-bias MAC (Wmax <= 64); iq4_xs (Wmax 127) keeps the sign trick
// q4_K / q5_K: BIAS = 0
 ಮಾದರಿ ಶೂನ್ಯ tiled_ಸತತ_microtile<32, ನಿಜ, 0, ಸುಳ್ಳು>(ಸ್ಥಿರಾಂಕ tiled_tile_src0 & src0, ಸ್ಥಿರಾಂಕ tiled_tile_src1 & src1,
                                               ಪೂರ್ಣಾಂಕ i0, ಪೂರ್ಣಾಂಕ j0, ಪೂರ್ಣಾಂಕ ಅಂಕೆ_k, ಪೂರ್ಣಾಂಕ slab, ದಶಮಾಂಶ * buf, ಪೂರ್ಣಾಂಕ buf_stride);
// iq4_xs: sign trick (Wmax 127 overflows act-bias)
ಮಾದರಿ ಶೂನ್ಯ tiled_ಸತತ_microtile<32, ಸುಳ್ಳು, 128, ಸುಳ್ಳು>(ಸ್ಥಿರಾಂಕ tiled_tile_src0 & src0, ಸ್ಥಿರಾಂಕ tiled_tile_src1 & src1,
                                                  ಪೂರ್ಣಾಂಕ i0, ಪೂರ್ಣಾಂಕ j0, ಪೂರ್ಣಾಂಕ ಅಂಕೆ_k, ಪೂರ್ಣಾಂಕ slab, ದಶಮಾಂಶ * buf, ಪೂರ್ಣಾಂಕ buf_stride);
// iq2_xxs, iq3_xxs, iq3_s, iq1_s: act-bias (Wmax <= 62)
ಮಾದರಿ ಶೂನ್ಯ tiled_ಸತತ_microtile<32, ಸುಳ್ಳು, 128, ನಿಜ>(ಸ್ಥಿರಾಂಕ tiled_tile_src0 & src0, ಸ್ಥಿರಾಂಕ tiled_tile_src1 & src1,
                                                  ಪೂರ್ಣಾಂಕ i0, ಪೂರ್ಣಾಂಕ j0, ಪೂರ್ಣಾಂಕ ಅಂಕೆ_k, ಪೂರ್ಣಾಂಕ slab, ದಶಮಾಂಶ * buf, ಪೂರ್ಣಾಂಕ buf_stride);
// iq2_xs, iq2_s, iq1_m: act-bias
ಮಾದರಿ ಶೂನ್ಯ tiled_ಸತತ_microtile<16, ಸುಳ್ಳು, 128, ನಿಜ>(ಸ್ಥಿರಾಂಕ tiled_tile_src0 & src0, ಸ್ಥಿರಾಂಕ tiled_tile_src1 & src1,
                                                  ಪೂರ್ಣಾಂಕ i0, ಪೂರ್ಣಾಂಕ j0, ಪೂರ್ಣಾಂಕ ಅಂಕೆ_k, ಪೂರ್ಣಾಂಕ slab, ದಶಮಾಂಶ * buf, ಪೂರ್ಣಾಂಕ buf_stride);
// q6_K: act-bias (Wmax 32)
ಮಾದರಿ ಶೂನ್ಯ tiled_ಸತತ_microtile<16, ಸುಳ್ಳು, 32, ನಿಜ>(ಸ್ಥಿರಾಂಕ tiled_tile_src0 & src0, ಸ್ಥಿರಾಂಕ tiled_tile_src1 & src1,
                                                 ಪೂರ್ಣಾಂಕ i0, ಪೂರ್ಣಾಂಕ j0, ಪೂರ್ಣಾಂಕ ಅಂಕೆ_k, ಪೂರ್ಣಾಂಕ slab, ದಶಮಾಂಶ * buf, ಪೂರ್ಣಾಂಕ buf_stride);
// q3_K: act-bias (Wmax 4)
ಮಾದರಿ ಶೂನ್ಯ tiled_ಸತತ_microtile<16, ಸುಳ್ಳು, 4, ನಿಜ>(ಸ್ಥಿರಾಂಕ tiled_tile_src0 & src0, ಸ್ಥಿರಾಂಕ tiled_tile_src1 & src1,
                                                ಪೂರ್ಣಾಂಕ i0, ಪೂರ್ಣಾಂಕ j0, ಪೂರ್ಣಾಂಕ ಅಂಕೆ_k, ಪೂರ್ಣಾಂಕ slab, ದಶಮಾಂಶ * buf, ಪೂರ್ಣಾಂಕ buf_stride);
// q2_K: BIAS = 0
ಮಾದರಿ ಶೂನ್ಯ tiled_ಸತತ_microtile<16, ನಿಜ, 0, ಸುಳ್ಳು>(ಸ್ಥಿರಾಂಕ tiled_tile_src0 & src0, ಸ್ಥಿರಾಂಕ tiled_tile_src1 & src1,
                                               ಪೂರ್ಣಾಂಕ i0, ಪೂರ್ಣಾಂಕ j0, ಪೂರ್ಣಾಂಕ ಅಂಕೆ_k, ಪೂರ್ಣಾಂಕ slab, ದಶಮಾಂಶ * buf, ಪೂರ್ಣಾಂಕ buf_stride);


#ವ್ಯಾಖ್ಯಾನಿಸು MIN(a, b) ((a) < (b) ? (a) : (b))

// block_q8_K in 4-byte words, for the int32 gather indices
static_assert(ಗಾತ್ರ(ಖಂಡ_q8_K) == 292 && offsetof(ಖಂಡ_q8_K, qs) == 4,
              "block_q8_K layout changed, fix the src1 repack");

// VNNI: interleave the natural [row][k] codes in-place into the group-local
// [kg%16][kg/16][row][4] layout. base points at row 0 (row r at base + r*row_stride);
// n_tiles 16x16 int32 tiles are transposed to the right (tile t occupies bytes
// [t*64, t*64+64) per row). Each 16-row group is self-contained in row_stride*16 bytes.
#ಆದರೆ defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)
ಶೂನ್ಯ tiled_repack_src1(tiled_tile_src1 * src1, ಪೂರ್ಣಾಂಕ row0, ಪೂರ್ಣಾಂಕ ಅಂಕೆ_k, ತಾರ್ಕಿಕ bias) {
    GGML_ಬಳಸದ(bias);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_tiles = ಅಂಕೆ_k * (TILED_TILE_K / (TILED_MICRO * 4));
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು_stride = ಅಂಕೆ_k * TILED_TILE_K;
    uint8_t * ತಳ = (uint8_t *) src1->q + row0 * ಅಡ್ಡಸಾಲು_stride;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < n_tiles; t++) {
        uint8_t * cbase = ತಳ + t * (TILED_MICRO * 4);
        __m512i v[16];
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ r = 0; r < 16; r++) {
            v[r] = mm512_ತುಂಬು_si512((ಸ್ಥಿರಾಂಕ __m512i *) (cbase + r * ಅಡ್ಡಸಾಲು_stride));
        }

        // 16x16 int32 transpose, 4 butterfly phases
        // Phase 1: 1-element interleave, pairs (0,1), (2,3), ..., (14,15)
        {
            ಸ್ಥಿರಾಂಕ __m512i idx_a = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
            ಸ್ಥಿರಾಂಕ __m512i idx_b = _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 16; i += 2) {
                __m512i a = v[i], b = v[i+1];
                v[i]   = _mm512_permutex2var_epi32(a, idx_a, b);
                v[i+1] = _mm512_permutex2var_epi32(a, idx_b, b);
            }
        }
        // Phase 2: 2-element interleave, pairs (0,2), (1,3), (4,6), (5,7), ...
        {
            ಸ್ಥಿರಾಂಕ __m512i idx_a = _mm512_setr_epi32(0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29);
            ಸ್ಥಿರಾಂಕ __m512i idx_b = _mm512_setr_epi32(2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 16; i += 4) {
                __m512i a = v[i],   b = v[i+2];
                v[i]   = _mm512_permutex2var_epi32(a, idx_a, b);
                v[i+2] = _mm512_permutex2var_epi32(a, idx_b, b);
                a = v[i+1], b = v[i+3];
                v[i+1] = _mm512_permutex2var_epi32(a, idx_a, b);
                v[i+3] = _mm512_permutex2var_epi32(a, idx_b, b);
            }
        }
        // Phase 3: 4-element interleave, pairs (0,4), (1,5), ..., (7,11), (8,12), ...
        {
            ಸ್ಥಿರಾಂಕ __m512i idx_a = _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23);
            ಸ್ಥಿರಾಂಕ __m512i idx_b = _mm512_setr_epi32(8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 16; i += 8) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 4; j++) {
                    __m512i a = v[i+j], b = v[i+4+j];
                    v[i+j]   = _mm512_permutex2var_epi32(a, idx_a, b);
                    v[i+4+j] = _mm512_permutex2var_epi32(a, idx_b, b);
                }
            }
        }
        // Phase 4: 8-element interleave, pairs (0,8), (1,9), ..., (7,15)
        {
            ಸ್ಥಿರಾಂಕ __m512i idx_a = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
            ಸ್ಥಿರಾಂಕ __m512i idx_b = _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 8; i++) {
                __m512i a = v[i], b = v[i+8];
                v[i]   = _mm512_permutex2var_epi32(a, idx_a, b);
                v[i+8] = _mm512_permutex2var_epi32(a, idx_b, b);
            }
        }

        // store: v[g] holds 16 int32s for k-group col_order[g], rows 0..15
        ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_ಕ್ರಮ[16] = {0, 8, 1, 9, 4, 12, 5, 13, 2, 10, 3, 11, 6, 14, 7, 15};
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ g = 0; g < 16; g++) {
            _mm512_store_si512((ಶೂನ್ಯ *) (cbase + ಲಂಬಸಾಲು_ಕ್ರಮ[g] * ಅಡ್ಡಸಾಲು_stride), v[g]);
        }
    }
}
#ಅಥವಾಆದರೆ defined(__AVX2__)
// act-bias: pre-bias the activation in place (+128) so the MAC's maddubs(q1b, w) sees unsigned bytes;
// no transpose on AVX2 (natural layout). 16 rows x n_tiles*64 bytes, row r at base + r*row_stride
ಶೂನ್ಯ tiled_repack_src1(tiled_tile_src1 * src1, ಪೂರ್ಣಾಂಕ row0, ಪೂರ್ಣಾಂಕ ಅಂಕೆ_k, ತಾರ್ಕಿಕ bias) {
    ಆದರೆ (!bias) {
        ಹಿಂತಿರುಗಿ;
    }
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು_stride = ಅಂಕೆ_k * TILED_TILE_K;
    uint8_t * ತಳ = (uint8_t *) src1->q + row0 * ಅಡ್ಡಸಾಲು_stride;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nbytes = ಅಡ್ಡಸಾಲು_stride; // full row width
    ಸ್ಥಿರಾಂಕ __m256i b128 = _mm256_set1_epi8((int8_t) 0x80);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ r = 0; r < TILED_MICRO; r++) {
        uint8_t * ಅಡ್ಡಸಾಲು = ತಳ + r * ಅಡ್ಡಸಾಲು_stride;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ off = 0; off < nbytes; off += 32) {
            ಸ್ಥಿರಾಂಕ __m256i v = _mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *) (ಅಡ್ಡಸಾಲು + off));
            _mm256_storeu_si256((__m256i *) (ಅಡ್ಡಸಾಲು + off), _mm256_xor_si256(v, b128));
        }
    }
}
#ಇಲ್ಲದಿದ್ದರೆ
ಶೂನ್ಯ tiled_repack_src1(tiled_tile_src1 * src1, ಪೂರ್ಣಾಂಕ row0, ಪೂರ್ಣಾಂಕ ಅಂಕೆ_k, ತಾರ್ಕಿಕ bias) {
    GGML_ಬಳಸದ(src1); GGML_ಬಳಸದ(row0); GGML_ಬಳಸದ(ಅಂಕೆ_k); GGML_ಬಳಸದ(bias);
}
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__AVX2__)
// sum 32 unsigned bytes to int32 (via maddubs -> int16 pairs -> hsum)
ಸ್ಥಿರ int32_t tiled_byte_ಮೊತ್ತ_32(ಸ್ಥಿರಾಂಕ uint8_t * p) {
    ಸ್ಥಿರಾಂಕ __m256i dot = _mm256_maddubs_epi16(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *) p), _mm256_set1_epi8(1));
    ಸ್ಥಿರಾಂಕ __m256i ಮೊತ್ತ = _mm256_madd_epi16(_mm256_set1_epi16(1), dot); // 8 int32
    // reduce 8 lanes: lo+hi -> 4 lanes, hadd -> 2, hadd -> 1 (a single hadd pair only folds 4 lanes)
    ಸ್ಥಿರಾಂಕ __m128i t = mm_ಕೂಡಿಸು_epi32(_mm256_castsi256_si128(ಮೊತ್ತ), _mm256_extracti128_si256(ಮೊತ್ತ, 1));
    ಸ್ಥಿರಾಂಕ __m128i h = _mm_hadd_epi32(t, t);
    ಹಿಂತಿರುಗಿ _mm_cvtsi128_si32(_mm_hadd_epi32(h, h));
}
// sum 16 unsigned bytes to int32
ಸ್ಥಿರ int32_t tiled_byte_ಮೊತ್ತ_16(ಸ್ಥಿರಾಂಕ uint8_t * p) {
    ಸ್ಥಿರಾಂಕ __m128i dot = _mm_maddubs_epi16(_mm_loadu_si128((ಸ್ಥಿರಾಂಕ __m128i *) p), _mm_set1_epi8(1));
    ಸ್ಥಿರಾಂಕ __m128i ಮೊತ್ತ = _mm_madd_epi16(_mm_set1_epi16(1), dot); // 4 int32
    ಸ್ಥಿರಾಂಕ __m128i h = _mm_hadd_epi32(ಮೊತ್ತ, ಮೊತ್ತ);
    ಹಿಂತಿರುಗಿ _mm_cvtsi128_si32(_mm_hadd_epi32(h, h));
}
#ಆದರೆ_ಅಂತ್ಯ

// act-bias: precompute the per-weight-row correction corr = 128*sum_s scales[s]*w_bsum[s] where
// w_bsum[s] = sum of the debiased weight bytes in subblock s. Stored in mins[r][0] (unused for
// HAS_MIN = false, which is exactly the act-bias case). Reused across all act bands and weight groups.
// No-op off AVX2 (the act-bias MAC is AVX2-only) and when corr is false (iq4_xs / BIAS = 0).
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ SUBBLK>
ಶೂನ್ಯ tiled_repack_src0(tiled_tile_src0 * tile, ಪೂರ್ಣಾಂಕ n_ಅಡ್ಡಸಾಲುಗಳು, ಪೂರ್ಣಾಂಕ ಅಂಕೆ_k, ಪೂರ್ಣಾಂಕ BIAS, ತಾರ್ಕಿಕ corr) {
#ಆದರೆ defined(__AVX2__)
    ಆದರೆ (!corr) {
        ಹಿಂತಿರುಗಿ;
    }
    constexpr ಪೂರ್ಣಾಂಕ NB = TILED_TILE_K / SUBBLK;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb_stride = NB * ಅಂಕೆ_k;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ slab = 0; slab < ಅಂಕೆ_k; slab++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ r = 0; r < n_ಅಡ್ಡಸಾಲುಗಳು; r++) {
            ಸ್ಥಿರಾಂಕ uint8_t * q = &tile->q[r * (ಅಂಕೆ_k * TILED_TILE_K) + slab * TILED_TILE_K];
            ಸ್ಥಿರಾಂಕ int32_t * scales = &tile->scales[r * nb_stride + slab * NB];
            int32_t corr_val = 0;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 0; s < NB; s++) {
                ಸ್ಥಿರಾಂಕ int32_t qsum = (SUBBLK == 32) ? tiled_byte_ಮೊತ್ತ_32(&q[s * SUBBLK]) : tiled_byte_ಮೊತ್ತ_16(&q[s * SUBBLK]);
                corr_val += 128 * scales[s] * (qsum - SUBBLK * BIAS);
            }
            tile->mins[r * nb_stride + slab * NB] = corr_val;
        }
    }
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ಬಳಸದ(tile); GGML_ಬಳಸದ(n_ಅಡ್ಡಸಾಲುಗಳು); GGML_ಬಳಸದ(ಅಂಕೆ_k); GGML_ಬಳಸದ(BIAS); GGML_ಬಳಸದ(corr);
#ಆದರೆ_ಅಂತ್ಯ
}

ಮಾದರಿ ಶೂನ್ಯ tiled_repack_src0<16>(tiled_tile_src0 * tile, ಪೂರ್ಣಾಂಕ n_ಅಡ್ಡಸಾಲುಗಳು, ಪೂರ್ಣಾಂಕ ಅಂಕೆ_k, ಪೂರ್ಣಾಂಕ BIAS, ತಾರ್ಕಿಕ corr);
ಮಾದರಿ ಶೂನ್ಯ tiled_repack_src0<32>(tiled_tile_src0 * tile, ಪೂರ್ಣಾಂಕ n_ಅಡ್ಡಸಾಲುಗಳು, ಪೂರ್ಣಾಂಕ ಅಂಕೆ_k, ಪೂರ್ಣಾಂಕ BIAS, ತಾರ್ಕಿಕ corr);


