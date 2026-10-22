// mulmat microtile kernels

#ଅନ୍ତର୍ଭାଵ "tiled-kernel.h"

#ଅନ୍ତର୍ଭାଵ "ggml.h"

#ଅନ୍ତର୍ଭାଵ <cstring>

#ଯଦି defined(__AVX512VNNI__) || defined(__AVX2__) || defined(__AVX__)
#ଅନ୍ତର୍ଭାଵ <immintrin.h>
#endif


// Reference implementation, slower than existing vec_dot approach
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା SUBBLK, ତାର୍କିକ HAS_MIN, ପୂର୍ଣ୍ଣସଂଖ୍ୟା BIAS, ପୂର୍ଣ୍ଣସଂଖ୍ୟା NK>
ସ୍ଥିର ଶୂନ୍ୟ tiled_run_microtile_scalar(ଅଚଳ tiled_tile_src0 & src0, ଅଚଳ tiled_tile_src1 & src1,
                                       ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା j0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_k, ପୂର୍ଣ୍ଣସଂଖ୍ୟା slab, ଦଶମିକ * buf, ପୂର୍ଣ୍ଣସଂଖ୍ୟା buf_stride) {
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା NB = TILED_TILE_K / SUBBLK;    // subblocks per 256-K block
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା NS = SUBBLK / 16; // per-16 bsums per subblock

    // num_k/slab: the narrow path holds num_k slabs at row stride num_k*256 (num_k=1, slab=0 = standard)
    // NK: 1 = standard single-slab (offsets 0, strides compile-time); 0 = runtime
    // (num_k/slab from the args). The narrow path is memory-bound, so one runtime
    // version serves all of it.
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nkr  = (NK > 0) ? NK : ଅଙ୍କ_k;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା seff = (NK > 0) ? 0 : slab;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk_stride = nkr * TILED_TILE_K;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk_off = seff * TILED_TILE_K;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb_stride = NB * nkr;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb_off = seff * NB;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା bs_stride = (nkr == 1) ? TILED_TILE_ROWS : TILED_MICRO;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା bs_off = seff * TILED_MICRO;

    ଦଶମିକ acc[TILED_MICRO][TILED_MICRO];
    memset(acc, 0, ଆକାର(acc));

    // subdots at subblock granularity over the 256-K block, exact integer math
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = 0; s < NB; s++) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < TILED_MICRO; j++) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା br = j0 + j;
            ଅଚଳ int8_t * q1 = &src1.q[br * qk_stride + qk_off + s * SUBBLK];
            int32_t bsum = 0;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା u = 0; u < NS; u++) {
                bsum += src1.bsums[(bs_off + s * NS + u) * bs_stride + br];
            }

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < TILED_MICRO; i++) {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ar = i0 + i;
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା d_off = seff * TILED_MICRO + ar;
                ଅଚଳ uint8_t * q0 = &src0.q[ar * qk_stride + qk_off + s * SUBBLK];

                int32_t raw = 0;
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା e = 0; e < SUBBLK; e++) {
                    raw += (int32_t) q0[e] * (int32_t) q1[e];
                }

                // BIAS: subtract BIAS*bsum (src1's per-subblock code sum) from the exact int raw
                int32_t corr = raw;
                ଯଦି constexpr (BIAS != 0) {
                    corr -= BIAS * bsum;
                }
                ଅଚଳ int32_t scales_raw = (int32_t) src0.scales[ar * nb_stride + nb_off + s] * corr;
                // d is NOT applied here: it is constant over the s-loop, so we apply it last before write-out
                ଯଦି constexpr (HAS_MIN) {
                    ଅଚଳ int32_t mins_bsum = (int32_t) src0.mins[ar * nb_stride + nb_off + s] * bsum;
                    acc[i][j] += (ଦଶମିକ) src0.d[d_off] * (ଦଶମିକ) scales_raw
                              - (ଦଶମିକ) src0.dmin[d_off] * (ଦଶମିକ) mins_bsum;
                } ଅନ୍ୟଥା {
                    acc[i][j] += (ଦଶମିକ) src0.d[d_off] * (ଦଶମିକ) scales_raw;
                }
            }
        }
    }

    // Apply d and write out to buf
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < TILED_MICRO; i++) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < TILED_MICRO; j++) {
            buf[(i0 + i) * buf_stride + (j0 + j)] += src1.d[seff * TILED_MICRO + j0 + j] * acc[i][j];
        }
    }
}

#ଯଦି defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)


// VNNI microkernel
// 8x16 band pass: 8 src0 rows (i0, i0+8) x 16 src1 cols (j0, j0+16)
// Math: s1_acc += scales_s*raw_s, s2_acc += mins_s*bsums_s per subblock; the row result
// is d0*s1_acc - dmin*s2_acc (f32, exact in range), applied once per row.
//
// Register pressure: the band pass holds acc16 + s1_acc = 16 zmm for the
// 8-row band (s2_acc has no dpbusd dependency, so it is computed after the
// band pass and adds no register pressure);
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା SUBBLK, ତାର୍କିକ HAS_MIN, ପୂର୍ଣ୍ଣସଂଖ୍ୟା BIAS, ପୂର୍ଣ୍ଣସଂଖ୍ୟା NK>
ସ୍ଥିର ଶୂନ୍ୟ tiled_run_micro_vnni_8x16(ଅଚଳ tiled_tile_src0 & src0, ଅଚଳ tiled_tile_src1 & src1,
                                          ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା j0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_k, ପୂର୍ଣ୍ଣସଂଖ୍ୟା slab, ଦଶମିକ * buf, ପୂର୍ଣ୍ଣସଂଖ୍ୟା buf_stride) {
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା NB = TILED_TILE_K / SUBBLK;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା NS = SUBBLK / 16;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା NG = SUBBLK / 4;

    // band width, see the register-pressure note above
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_ROWS = 8;

    // num_k = K-blocks per row (the narrow path holds num_k slabs at row stride num_k*256 so
    // each weight row is one long stream); slab = this call's slab. num_k=1, slab=0 = standard.
    // NK: 1 = standard single-slab (offsets 0, strides compile-time); 0 = runtime
    // (num_k/slab from the args). The narrow path is memory-bound, so one runtime
    // version serves all of it.
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nkr  = (NK > 0) ? NK : ଅଙ୍କ_k;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା seff = (NK > 0) ? 0 : slab;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk_stride = nkr * TILED_TILE_K;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk_off = seff * TILED_TILE_K;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb_stride = NB * nkr;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb_off = seff * NB;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା bs_stride = (nkr == 1) ? TILED_TILE_ROWS : TILED_MICRO;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା bs_off = seff * TILED_MICRO;

    ଅଚଳ __m512 d1_vec = _mm512_load_ps(&src1.d[seff * TILED_MICRO + j0]);

    __m512i s1_acc[ଅଙ୍କ_ROWS];
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = 0; t < ଅଙ୍କ_ROWS; t++) { s1_acc[t] = _mm512_setzero_si512(); }
    __m512i s2_acc[ଅଙ୍କ_ROWS];
    ଯଦି constexpr (HAS_MIN) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = 0; t < ଅଙ୍କ_ROWS; t++) { s2_acc[t] = _mm512_setzero_si512(); }
    }

    ଅଚଳ int32_t * bsums = src1.bsums; // int32 per-16 sums; NS > 1 combines NS lanes per subblock
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = 0; s < NB; s++) {
        __m512i bsums32 = _mm512_load_si512((ଅଚଳ __m512i *) &bsums[(bs_off + s * NS) * bs_stride + j0]);
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା u = 1; u < NS; u++) {
            bsums32 = _mm512_add_epi32(bsums32, _mm512_load_si512((ଅଚଳ __m512i *) &bsums[(bs_off + s * NS + u) * bs_stride + j0]));
        }

        __m512i bias32 = _mm512_setzero_si512();
        ଯଦି constexpr (BIAS != 0) {
            bias32 = _mm512_mullo_epi32(bsums32, _mm512_set1_epi32(BIAS));
        }

        __m512i acc16[ଅଙ୍କ_ROWS];
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = 0; t < ଅଙ୍କ_ROWS; t++) { acc16[t] = _mm512_setzero_si512(); }

#ifdef __GNUC__
        #pragma GCC unroll 8 // pragma unrolled justified by measuing with/without
#endif
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା g = 0; g < NG; g++) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା kg = s * NG + g;
            // in-place interleave layout: [kg%16 @ TILED_TILE_K][kg/16 @ 64][row @ 4]
            ଅଚଳ __m512i codes = _mm512_load_si512((ଅଚଳ __m512i *) &src1.q[(j0 / TILED_MICRO) * (TILED_MICRO * qk_stride)
                + qk_off + (kg % TILED_MICRO) * qk_stride + (kg / TILED_MICRO) * (TILED_MICRO * 4)]);
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = 0; t < ଅଙ୍କ_ROWS; t++) {
                ଅଚଳ uint32_t u4 = *(ଅଚଳ uint32_t *) &src0.q[(i0 + t) * qk_stride + qk_off + kg * 4];
                ଅଚଳ __m512i u4b = _mm512_set1_epi32((ପୂର୍ଣ୍ଣସଂଖ୍ୟା) u4);
                acc16[t] = _mm512_dpbusd_epi32(acc16[t], u4b, codes);
            }
        }

        // int correction: s1_acc += scales*(raw-BIAS*bsums)
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = 0; t < ଅଙ୍କ_ROWS; t++) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ar = i0 + t;
            __m512i rawi = acc16[t];
            ଯଦି constexpr (BIAS != 0) {
                rawi = _mm512_sub_epi32(rawi, bias32);
            }
            s1_acc[t] = _mm512_add_epi32(s1_acc[t],
                _mm512_mullo_epi32(rawi, _mm512_set1_epi32(src0.scales[ar * nb_stride + nb_off + s])));
        }
    }

    // s2_acc += mins*bsums; independent of the dpbusd results, so it runs here instead of in the
    // band pass: the band pass keeps the register budget for the 8-row band
    ଯଦି constexpr (HAS_MIN) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = 0; s < NB; s++) {
            __m512i bsums32 = _mm512_load_si512((ଅଚଳ __m512i *) &bsums[(bs_off + s * NS) * bs_stride + j0]);
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା u = 1; u < NS; u++) {
                bsums32 = _mm512_add_epi32(bsums32, _mm512_load_si512((ଅଚଳ __m512i *) &bsums[(bs_off + s * NS + u) * bs_stride + j0]));
            }
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = 0; t < ଅଙ୍କ_ROWS; t++) {
                s2_acc[t] = _mm512_add_epi32(s2_acc[t],
                    _mm512_mullo_epi32(bsums32, _mm512_set1_epi32(src0.mins[(i0 + t) * nb_stride + nb_off + s])));
            }
        }
    }

    // epilogue: int->float, apply per-row scales, store to buf
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = 0; t < ଅଙ୍କ_ROWS; t++) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ar = i0 + t;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା d_off = seff * TILED_MICRO + ar;
        __m512 f1 = _mm512_cvtepi32_ps(s1_acc[t]);
        __m512 result = _mm512_mul_ps(f1, _mm512_set1_ps(src0.d[d_off]));
        ଯଦି constexpr (HAS_MIN) {
            __m512 f2 = _mm512_cvtepi32_ps(s2_acc[t]);
            result = _mm512_fnmadd_ps(_mm512_set1_ps(src0.dmin[d_off]), f2, result);
        }
        ଦଶମିକ * p = &buf[(i0 + t) * buf_stride + j0];
        _mm512_store_ps(p, _mm512_add_ps(_mm512_load_ps(p), _mm512_mul_ps(result, d1_vec)));
    }
}

// 16x16 microtile as two explicit 8x16 band passes
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା SUBBLK, ତାର୍କିକ HAS_MIN, ପୂର୍ଣ୍ଣସଂଖ୍ୟା BIAS, ପୂର୍ଣ୍ଣସଂଖ୍ୟା NK>
ସ୍ଥିର ଶୂନ୍ୟ tiled_run_microtile_vnni(ଅଚଳ tiled_tile_src0 & src0, ଅଚଳ tiled_tile_src1 & src1,
                                         ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା j0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_k, ପୂର୍ଣ୍ଣସଂଖ୍ୟା slab, ଦଶମିକ * buf, ପୂର୍ଣ୍ଣସଂଖ୍ୟା buf_stride) {
    tiled_run_micro_vnni_8x16<SUBBLK, HAS_MIN, BIAS, NK>(src0, src1, i0,      j0, ଅଙ୍କ_k, slab, buf, buf_stride);
    tiled_run_micro_vnni_8x16<SUBBLK, HAS_MIN, BIAS, NK>(src0, src1, i0 + 8,  j0, ଅଙ୍କ_k, slab, buf, buf_stride);
}

#endif // __AVX512VNNI__ && __AVX512VL__

#ଯଦି defined(__AVX2__)

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
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା SUBBLK, ତାର୍କିକ HAS_MIN, ପୂର୍ଣ୍ଣସଂଖ୍ୟା BIAS, ପୂର୍ଣ୍ଣସଂଖ୍ୟା NK, ତାର୍କିକ ACTBIAS>
ସ୍ଥିର ଶୂନ୍ୟ tiled_run_microtile_avx2(ଅଚଳ tiled_tile_src0 & src0, ଅଚଳ tiled_tile_src1 & src1,
                                     ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା j0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_k, ପୂର୍ଣ୍ଣସଂଖ୍ୟା slab, ଦଶମିକ * buf, ପୂର୍ଣ୍ଣସଂଖ୍ୟା buf_stride) {
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା NB = TILED_TILE_K / SUBBLK;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା GROUP = 8; // Process 8 src1 columns concurrently (1 YMM vector)

    static_assert(SUBBLK == 16 || SUBBLK == 32, "unsupported SUBBLK");

    // num_k/slab: the narrow path holds num_k slabs at row stride num_k*256 (num_k=1, slab=0 = standard)
    // NK: 1 = standard single-slab (offsets 0, strides compile-time); 0 = runtime
    // (num_k/slab from the args). The narrow path is memory-bound, so one runtime
    // version serves all of it.
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nkr  = (NK > 0) ? NK : ଅଙ୍କ_k;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା seff = (NK > 0) ? 0 : slab;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk_stride = nkr * TILED_TILE_K;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk_off = seff * TILED_TILE_K;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb_stride = NB * nkr;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb_off = seff * NB;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା bs_stride = (nkr == 1) ? TILED_TILE_ROWS : TILED_MICRO;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା bs_off = seff * TILED_MICRO;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < TILED_MICRO; i++) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ar = i0 + i;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା d_off = seff * TILED_MICRO + ar;
        ଅଚଳ ଦଶମିକ d0 = src0.d[d_off];
        ଅଚଳ ଦଶମିକ dmin0 = src0.dmin[d_off];
        ଅଚଳ uint8_t * q0 = &src0.q[ar * qk_stride + qk_off];
        ଅଚଳ int32_t * scales_row = &src0.scales[ar * nb_stride + nb_off];
        ଅଚଳ int32_t * mins_row = &src0.mins[ar * nb_stride + nb_off];

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା g = 0; g < TILED_MICRO; g += GROUP) {
            // Pre-calculate weight pointers for this column group
            ଅଚଳ int8_t * q1_ptr[GROUP];
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = 0; t < GROUP; t++) {
                q1_ptr[t] = &src1.q[(j0 + g + t) * qk_stride + qk_off];
            }

            __m256i s2   = _mm256_setzero_si256();

            __m256i acc[GROUP];
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = 0; t < GROUP; t++) {
                acc[t] = _mm256_setzero_si256();
            }

            ଯଦି constexpr (SUBBLK == 32) {
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = 0; s < NB; s++) {
                    // OPTIMIZATION 1: Use aligned loads (_mm256_load_si256)
                    ଅଚଳ __m256i q0_32 = _mm256_load_si256((ଅଚଳ __m256i *) &q0[s * SUBBLK]);
                    ଅଚଳ __m256i scales16 = _mm256_set1_epi16(scales_row[s]);

                    ଯଦି constexpr (BIAS != 0) {
                        ଅଚଳ __m256i w = _mm256_sub_epi8(q0_32, _mm256_set1_epi8((int8_t) BIAS)); // debiased weight
                        ଯଦି constexpr (ACTBIAS) {
                            // activation pre-biased (+128) by repack; q1b unsigned, w signed
                            #pragma GCC unroll 8
                            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = 0; t < GROUP; t++) {
                                ଅଚଳ __m256i q1b_32 = _mm256_load_si256((ଅଚଳ __m256i *) &q1_ptr[t][s * SUBBLK]);
                                acc[t] = _mm256_add_epi32(acc[t],
                                    _mm256_madd_epi16(scales16, _mm256_maddubs_epi16(q1b_32, w)));
                            }
                        } ଅନ୍ୟଥା {
                            // iq4_xs (Wmax > 64 overflows act-bias): sign trick, both operands <= 127
                            ଅଚଳ __m256i ax = _mm256_sign_epi8(w, w);
                            #pragma GCC unroll 8
                            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = 0; t < GROUP; t++) {
                                ଅଚଳ __m256i q1_32 = _mm256_load_si256((ଅଚଳ __m256i *) &q1_ptr[t][s * SUBBLK]);
                                acc[t] = _mm256_add_epi32(acc[t],
                                    _mm256_madd_epi16(scales16, _mm256_maddubs_epi16(ax, _mm256_sign_epi8(q1_32, w))));
                            }
                        }
                    } ଅନ୍ୟଥା {
                        #pragma GCC unroll 8
                        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = 0; t < GROUP; t++) {
                            ଅଚଳ __m256i q1_32 = _mm256_load_si256((ଅଚଳ __m256i *) &q1_ptr[t][s * SUBBLK]);
                            acc[t] = _mm256_add_epi32(acc[t],
                                _mm256_madd_epi16(scales16, _mm256_maddubs_epi16(q0_32, q1_32)));
                        }
                    }

                    ଯଦି constexpr (HAS_MIN) {
                        ଅଚଳ __m256i bsums_v = _mm256_add_epi32(
                            _mm256_load_si256((ଅଚଳ __m256i *) &src1.bsums[(bs_off + s * 2) * bs_stride + j0 + g]),
                            _mm256_load_si256((ଅଚଳ __m256i *) &src1.bsums[(bs_off + s * 2 + 1) * bs_stride + j0 + g]));
                        s2 = _mm256_add_epi32(s2, _mm256_mullo_epi32(bsums_v, _mm256_set1_epi32(mins_row[s])));
                    }
                }
            } ଅନ୍ୟଥା { // SUBBLK == 16
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା sp = 0; sp < NB; sp += 2) {
                    ଅଚଳ __m256i q0_32 = _mm256_load_si256((ଅଚଳ __m256i *) &q0[sp * SUBBLK]);
                    ଅଚଳ __m256i scalesv = _mm256_set_m128i(_mm_set1_epi16(scales_row[sp + 1]), _mm_set1_epi16(scales_row[sp]));
                    ଯଦି constexpr (BIAS != 0) {
                        ଅଚଳ __m256i w = _mm256_sub_epi8(q0_32, _mm256_set1_epi8((int8_t) BIAS)); // debiased weight
                        ଯଦି constexpr (ACTBIAS) {
                            // activation pre-biased (+128) by repack; q1b unsigned, w signed
                            #pragma GCC unroll 8
                            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = 0; t < GROUP; t++) {
                                ଅଚଳ __m256i q1b_32 = _mm256_load_si256((ଅଚଳ __m256i *) &q1_ptr[t][sp * SUBBLK]);
                                acc[t] = _mm256_add_epi32(acc[t], _mm256_madd_epi16(
                                    scalesv, _mm256_maddubs_epi16(q1b_32, w)));
                            }
                        } ଅନ୍ୟଥା {
                            // iq4_xs (Wmax > 64 overflows act-bias): sign trick, both operands <= 127
                            ଅଚଳ __m256i ax = _mm256_sign_epi8(w, w);
                            #pragma GCC unroll 8
                            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = 0; t < GROUP; t++) {
                                ଅଚଳ __m256i q1_32 = _mm256_load_si256((ଅଚଳ __m256i *) &q1_ptr[t][sp * SUBBLK]);
                                acc[t] = _mm256_add_epi32(acc[t], _mm256_madd_epi16(
                                    scalesv, _mm256_maddubs_epi16(ax, _mm256_sign_epi8(q1_32, w))));
                            }
                        }
                    } ଅନ୍ୟଥା {
                        #pragma GCC unroll 8
                        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = 0; t < GROUP; t++) {
                            ଅଚଳ __m256i q1_32 = _mm256_load_si256((ଅଚଳ __m256i *) &q1_ptr[t][sp * SUBBLK]);
                            acc[t] = _mm256_add_epi32(acc[t], _mm256_madd_epi16(
                                scalesv, _mm256_maddubs_epi16(q0_32, q1_32)));
                        }
                    }

                    ଯଦି constexpr (HAS_MIN) {
                        ଅଚଳ __m256i bsums0_v = _mm256_load_si256((ଅଚଳ __m256i *) &src1.bsums[(bs_off + sp) * bs_stride + j0 + g]);
                        ଅଚଳ __m256i bsums1_v = _mm256_load_si256((ଅଚଳ __m256i *) &src1.bsums[(bs_off + sp + 1) * bs_stride + j0 + g]);
                        s2 = _mm256_add_epi32(s2, _mm256_add_epi32(
                            _mm256_mullo_epi32(bsums0_v, _mm256_set1_epi32(mins_row[sp])),
                            _mm256_mullo_epi32(bsums1_v, _mm256_set1_epi32(mins_row[sp + 1]))));
                    }
                }
            }

            // Reduce the eight accumulators to column sums in acc[0]..acc[7] order, with additions modulo 2^32.
            ଅଚଳ __m256i ab = _mm256_hadd_epi32(acc[0], acc[1]);
            ଅଚଳ __m256i cd = _mm256_hadd_epi32(acc[2], acc[3]);
            ଅଚଳ __m256i ef = _mm256_hadd_epi32(acc[4], acc[5]);
            ଅଚଳ __m256i gh = _mm256_hadd_epi32(acc[6], acc[7]);

            // Each 128-bit half now holds four column sums over input lanes 0..3 or 4..7.
            ଅଚଳ __m256i abcd = _mm256_hadd_epi32(ab, cd);
            ଅଚଳ __m256i efgh = _mm256_hadd_epi32(ef, gh);
            ଅଚଳ __m256i lo = _mm256_permute2x128_si256(abcd, efgh, 0x20);
            ଅଚଳ __m256i hi = _mm256_permute2x128_si256(abcd, efgh, 0x31);
            __m256i s1_vec = _mm256_add_epi32(lo, hi);

            // OPTIMIZATION 2: Vectorized Epilogue with FMA3 gating
            ଯଦି constexpr (BIAS != 0 && ACTBIAS) {
                // act-bias correction: raw = desired + 128*sum_s scales[s]*w_bsum[s]; corr precomputed in repack_src0
                s1_vec = _mm256_sub_epi32(s1_vec, _mm256_set1_epi32(mins_row[0]));
            }

            ଅଚଳ __m256 src1_d_vec = _mm256_load_ps(&src1.d[seff * TILED_MICRO + j0 + g]);
            __m256 res_vec = _mm256_mul_ps(_mm256_set1_ps(d0), _mm256_cvtepi32_ps(s1_vec));

            ଯଦି constexpr (HAS_MIN) {
#ଯଦି defined(__FMA__)
                // FMA3 implementation: res_vec = res_vec - (dmin0 * s2)
                res_vec = _mm256_fnmadd_ps(_mm256_set1_ps(dmin0), _mm256_cvtepi32_ps(s2), res_vec);
#ଅନ୍ୟଥା
                // Non-FMA fallback
                res_vec = _mm256_sub_ps(res_vec, _mm256_mul_ps(_mm256_set1_ps(dmin0), _mm256_cvtepi32_ps(s2)));
#endif
            }

            ଦଶମିକ * buf_ptr = &buf[ar * buf_stride + j0 + g];
            __m256 current_buf = _mm256_loadu_ps(buf_ptr);

#ଯଦି defined(__FMA__)
            // FMA3 accumulation: current_buf + (res_vec * src1_d_vec)
            __m256 updated_buf = _mm256_fmadd_ps(res_vec, src1_d_vec, current_buf);
#ଅନ୍ୟଥା
            __m256 updated_buf = _mm256_add_ps(current_buf, _mm256_mul_ps(res_vec, src1_d_vec));
#endif

            _mm256_storeu_ps(buf_ptr, updated_buf);
        }
    }
}

#endif // __AVX2__

#ଯଦି defined(__AVX__) && !defined(__AVX2__)
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା SUBBLK, ତାର୍କିକ HAS_MIN, ପୂର୍ଣ୍ଣସଂଖ୍ୟା BIAS, ପୂର୍ଣ୍ଣସଂଖ୍ୟା NK>
ସ୍ଥିର ଶୂନ୍ୟ tiled_run_microtile_avx(ଅଚଳ tiled_tile_src0 & src0, ଅଚଳ tiled_tile_src1 & src1,
                                    ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା j0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_k, ପୂର୍ଣ୍ଣସଂଖ୍ୟା slab, ଦଶମିକ * buf, ପୂର୍ଣ୍ଣସଂଖ୍ୟା buf_stride) {
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା NB = TILED_TILE_K / SUBBLK;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା NS = SUBBLK / 16;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା GROUP = 4; // src1 columns per group: one acc32 per column

    // num_k/slab: the narrow path holds num_k slabs at row stride num_k*256 (num_k=1, slab=0 = standard)
    // NK: 1 = standard single-slab (offsets 0, strides compile-time); 0 = runtime
    // (num_k/slab from the args). The narrow path is memory-bound, so one runtime
    // version serves all of it.
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nkr  = (NK > 0) ? NK : ଅଙ୍କ_k;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା seff = (NK > 0) ? 0 : slab;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk_stride = nkr * TILED_TILE_K;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk_off = seff * TILED_TILE_K;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb_stride = NB * nkr;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb_off = seff * NB;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା bs_stride = (nkr == 1) ? TILED_TILE_ROWS : TILED_MICRO;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା bs_off = seff * TILED_MICRO;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < TILED_MICRO; i++) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ar = i0 + i;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା d_off = seff * TILED_MICRO + ar;
        ଅଚଳ ଦଶମିକ d0 = src0.d[d_off];
        ଅଚଳ ଦଶମିକ dmin0 = src0.dmin[d_off];
        ଅଚଳ uint8_t * q0 = &src0.q[ar * qk_stride + qk_off];
        ଅଚଳ int32_t * scales_row = &src0.scales[ar * nb_stride + nb_off];
        ଅଚଳ int32_t * mins_row = &src0.mins[ar * nb_stride + nb_off];

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା g = 0; g < TILED_MICRO; g += GROUP) {
            ଅଚଳ int8_t * q1g[GROUP];
            __m128i acc[GROUP];
            __m128i s2 = _mm_setzero_si128();   // sum_s mins_s * bsums_s
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = 0; t < GROUP; t++) {
                q1g[t] = &src1.q[(j0 + g + t) * qk_stride + qk_off];
                acc[t] = _mm_setzero_si128();
            }

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = 0; s < NB; s++) {
                ଅଚଳ __m128i scales16 = _mm_set1_epi16(scales_row[s]);
                ଯଦି constexpr (HAS_MIN) {
                    __m128i bsums_v = _mm_setzero_si128(); // per-col bsums as a 4-lane i32 vector
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା u = 0; u < NS; u++) {
                        bsums_v = _mm_add_epi32(bsums_v, _mm_loadu_si128(
                            (ଅଚଳ __m128i *) &src1.bsums[(bs_off + s * NS + u) * bs_stride + j0 + g]));
                    }
                    s2 = _mm_add_epi32(s2, _mm_mullo_epi32(bsums_v, _mm_set1_epi32(mins_row[s])));
                }
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା u = 0; u < NS; u++) {
                    ଅଚଳ __m128i a16 = _mm_loadu_si128((ଅଚଳ __m128i *) &q0[s * SUBBLK + u * 16]);
                    ଯଦି constexpr (BIAS != 0) {
                        // debiased weight w = a16 - BIAS; signed dot via the sign trick (see header)
                        ଅଚଳ __m128i a16s = _mm_sub_epi8(a16, _mm_set1_epi8((int8_t) BIAS));
                        ଅଚଳ __m128i ax    = _mm_sign_epi8(a16s, a16s);
                        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = 0; t < GROUP; t++) {
                            ଅଚଳ __m128i q1_16 = _mm_loadu_si128((ଅଚଳ __m128i *) &q1g[t][s * SUBBLK + u * 16]);
                            acc[t] = _mm_add_epi32(acc[t],
                                _mm_madd_epi16(scales16, _mm_maddubs_epi16(ax, _mm_sign_epi8(q1_16, a16s))));
                        }
                    } ଅନ୍ୟଥା {
                        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = 0; t < GROUP; t++) {
                            acc[t] = _mm_add_epi32(acc[t], _mm_madd_epi16(scales16,
                                _mm_maddubs_epi16(a16,
                                    _mm_loadu_si128((ଅଚଳ __m128i *) &q1g[t][s * SUBBLK + u * 16]))));
                        }
                    }
                }
            }

            int32_t s2_s[GROUP] = { 0, 0, 0, 0 };
            ଯଦି constexpr (HAS_MIN) {
                _mm_storeu_si128((__m128i *) s2_s, s2);
            }

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = 0; t < GROUP; t++) {
                // 4 i32 lanes -> scalar
                __m128i v = _mm_shuffle_epi32(acc[t], _MM_SHUFFLE(2, 3, 0, 1));
                acc[t] = _mm_add_epi32(acc[t], v);
                v = _mm_shuffle_epi32(acc[t], _MM_SHUFFLE(1, 0, 3, 2));
                acc[t] = _mm_add_epi32(acc[t], v);
                ଅଚଳ int32_t s1 = _mm_cvtsi128_si32(acc[t]);

                ଦଶମିକ res = d0 * (ଦଶମିକ) s1;
                ଯଦି constexpr (HAS_MIN) {
                    res -= dmin0 * (ଦଶମିକ) s2_s[t];
                }
                buf[ar * buf_stride + j0 + g + t] += src1.d[seff * TILED_MICRO + j0 + g + t] * res;
            }
        }
    }
}

#endif // __AVX__ && !__AVX2__

// main microtile entry point. num_k = K-blocks per row (the narrow path reads num_k slabs at row
// stride num_k*256 so each weight row is one long stream); slab = this call's slab. num_k=1,
// slab=0 is the standard single-slab path.
// num_k==1 dispatches to the NK=1 instantiation, whose strides/offsets are compile-time
// constants (shifts/scaled-leas). num_k>1 (the narrow path) uses the NK=0 runtime-strided
// version; it is memory-bound, so one version serves all of it.
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା SUBBLK, ତାର୍କିକ HAS_MIN, ପୂର୍ଣ୍ଣସଂଖ୍ୟା BIAS, ତାର୍କିକ ACTBIAS>
ଶୂନ୍ୟ tiled_run_microtile(ଅଚଳ tiled_tile_src0 & src0, ଅଚଳ tiled_tile_src1 & src1,
                         ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା j0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_k, ପୂର୍ଣ୍ଣସଂଖ୍ୟା slab, ଦଶମିକ * buf, ପୂର୍ଣ୍ଣସଂଖ୍ୟା buf_stride) {
    ଅଚଳ ତାର୍କିକ standard = (ଅଙ୍କ_k == 1);
#ଯଦି defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)
    ଯଦି (standard) {
        tiled_run_microtile_vnni<SUBBLK, HAS_MIN, BIAS, 1>(src0, src1, i0, j0, ଅଙ୍କ_k, slab, buf, buf_stride);
    } ଅନ୍ୟଥା {
        tiled_run_microtile_vnni<SUBBLK, HAS_MIN, BIAS, 0>(src0, src1, i0, j0, ଅଙ୍କ_k, slab, buf, buf_stride);
    }
#ନଚେତଯଦି defined(__AVX2__)
    ଯଦି (standard) {
        tiled_run_microtile_avx2<SUBBLK, HAS_MIN, BIAS, 1, ACTBIAS>(src0, src1, i0, j0, ଅଙ୍କ_k, slab, buf, buf_stride);
    } ଅନ୍ୟଥା {
        tiled_run_microtile_avx2<SUBBLK, HAS_MIN, BIAS, 0, ACTBIAS>(src0, src1, i0, j0, ଅଙ୍କ_k, slab, buf, buf_stride);
    }
#ନଚେତଯଦି defined(__AVX__)
    ଯଦି (standard) {
        tiled_run_microtile_avx<SUBBLK, HAS_MIN, BIAS, 1>(src0, src1, i0, j0, ଅଙ୍କ_k, slab, buf, buf_stride);
    } ଅନ୍ୟଥା {
        tiled_run_microtile_avx<SUBBLK, HAS_MIN, BIAS, 0>(src0, src1, i0, j0, ଅଙ୍କ_k, slab, buf, buf_stride);
    }
#ଅନ୍ୟଥା
    ଯଦି (standard) {
        tiled_run_microtile_scalar<SUBBLK, HAS_MIN, BIAS, 1>(src0, src1, i0, j0, ଅଙ୍କ_k, slab, buf, buf_stride);
    } ଅନ୍ୟଥା {
        tiled_run_microtile_scalar<SUBBLK, HAS_MIN, BIAS, 0>(src0, src1, i0, j0, ଅଙ୍କ_k, slab, buf, buf_stride);
    }
#endif
}

// explicit instantiations for the in-use formats (q4_K and q5_K share the constants)
// ACTBIAS selects the act-bias MAC (Wmax <= 64); iq4_xs (Wmax 127) keeps the sign trick
// q4_K / q5_K: BIAS = 0
 template ଶୂନ୍ୟ tiled_run_microtile<32, ସତ୍ୟ, 0, ମିଥ୍ୟା>(ଅଚଳ tiled_tile_src0 & src0, ଅଚଳ tiled_tile_src1 & src1,
                                               ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା j0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_k, ପୂର୍ଣ୍ଣସଂଖ୍ୟା slab, ଦଶମିକ * buf, ପୂର୍ଣ୍ଣସଂଖ୍ୟା buf_stride);
// iq4_xs: sign trick (Wmax 127 overflows act-bias)
template ଶୂନ୍ୟ tiled_run_microtile<32, ମିଥ୍ୟା, 128, ମିଥ୍ୟା>(ଅଚଳ tiled_tile_src0 & src0, ଅଚଳ tiled_tile_src1 & src1,
                                                  ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା j0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_k, ପୂର୍ଣ୍ଣସଂଖ୍ୟା slab, ଦଶମିକ * buf, ପୂର୍ଣ୍ଣସଂଖ୍ୟା buf_stride);
// iq2_xxs, iq3_xxs, iq3_s, iq1_s: act-bias (Wmax <= 62)
template ଶୂନ୍ୟ tiled_run_microtile<32, ମିଥ୍ୟା, 128, ସତ୍ୟ>(ଅଚଳ tiled_tile_src0 & src0, ଅଚଳ tiled_tile_src1 & src1,
                                                  ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା j0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_k, ପୂର୍ଣ୍ଣସଂଖ୍ୟା slab, ଦଶମିକ * buf, ପୂର୍ଣ୍ଣସଂଖ୍ୟା buf_stride);
// iq2_xs, iq2_s, iq1_m: act-bias
template ଶୂନ୍ୟ tiled_run_microtile<16, ମିଥ୍ୟା, 128, ସତ୍ୟ>(ଅଚଳ tiled_tile_src0 & src0, ଅଚଳ tiled_tile_src1 & src1,
                                                  ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା j0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_k, ପୂର୍ଣ୍ଣସଂଖ୍ୟା slab, ଦଶମିକ * buf, ପୂର୍ଣ୍ଣସଂଖ୍ୟା buf_stride);
// q6_K: act-bias (Wmax 32)
template ଶୂନ୍ୟ tiled_run_microtile<16, ମିଥ୍ୟା, 32, ସତ୍ୟ>(ଅଚଳ tiled_tile_src0 & src0, ଅଚଳ tiled_tile_src1 & src1,
                                                 ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା j0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_k, ପୂର୍ଣ୍ଣସଂଖ୍ୟା slab, ଦଶମିକ * buf, ପୂର୍ଣ୍ଣସଂଖ୍ୟା buf_stride);
// q3_K: act-bias (Wmax 4)
template ଶୂନ୍ୟ tiled_run_microtile<16, ମିଥ୍ୟା, 4, ସତ୍ୟ>(ଅଚଳ tiled_tile_src0 & src0, ଅଚଳ tiled_tile_src1 & src1,
                                                ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା j0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_k, ପୂର୍ଣ୍ଣସଂଖ୍ୟା slab, ଦଶମିକ * buf, ପୂର୍ଣ୍ଣସଂଖ୍ୟା buf_stride);
// q2_K: BIAS = 0
template ଶୂନ୍ୟ tiled_run_microtile<16, ସତ୍ୟ, 0, ମିଥ୍ୟା>(ଅଚଳ tiled_tile_src0 & src0, ଅଚଳ tiled_tile_src1 & src1,
                                               ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା j0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_k, ପୂର୍ଣ୍ଣସଂଖ୍ୟା slab, ଦଶମିକ * buf, ପୂର୍ଣ୍ଣସଂଖ୍ୟା buf_stride);


#define MIN(a, b) ((a) < (b) ? (a) : (b))

// block_q8_K in 4-byte words, for the int32 gather indices
static_assert(ଆକାର(block_q8_K) == 292 && offsetof(block_q8_K, qs) == 4,
              "block_q8_K layout changed, fix the src1 repack");

// VNNI: interleave the natural [row][k] codes in-place into the group-local
// [kg%16][kg/16][row][4] layout. base points at row 0 (row r at base + r*row_stride);
// n_tiles 16x16 int32 tiles are transposed to the right (tile t occupies bytes
// [t*64, t*64+64) per row). Each 16-row group is self-contained in row_stride*16 bytes.
#ଯଦି defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)
ଶୂନ୍ୟ tiled_repack_src1(tiled_tile_src1 * src1, ପୂର୍ଣ୍ଣସଂଖ୍ୟା row0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_k, ତାର୍କିକ bias) {
    GGML_UNUSED(bias);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_tiles = ଅଙ୍କ_k * (TILED_TILE_K / (TILED_MICRO * 4));
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row_stride = ଅଙ୍କ_k * TILED_TILE_K;
    uint8_t * base = (uint8_t *) src1->q + row0 * row_stride;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = 0; t < n_tiles; t++) {
        uint8_t * cbase = base + t * (TILED_MICRO * 4);
        __m512i v[16];
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା r = 0; r < 16; r++) {
            v[r] = _mm512_load_si512((ଅଚଳ __m512i *) (cbase + r * row_stride));
        }

        // 16x16 int32 transpose, 4 butterfly phases
        // Phase 1: 1-element interleave, pairs (0,1), (2,3), ..., (14,15)
        {
            ଅଚଳ __m512i idx_a = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
            ଅଚଳ __m512i idx_b = _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 16; i += 2) {
                __m512i a = v[i], b = v[i+1];
                v[i]   = _mm512_permutex2var_epi32(a, idx_a, b);
                v[i+1] = _mm512_permutex2var_epi32(a, idx_b, b);
            }
        }
        // Phase 2: 2-element interleave, pairs (0,2), (1,3), (4,6), (5,7), ...
        {
            ଅଚଳ __m512i idx_a = _mm512_setr_epi32(0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29);
            ଅଚଳ __m512i idx_b = _mm512_setr_epi32(2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31);
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 16; i += 4) {
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
            ଅଚଳ __m512i idx_a = _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23);
            ଅଚଳ __m512i idx_b = _mm512_setr_epi32(8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31);
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 16; i += 8) {
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < 4; j++) {
                    __m512i a = v[i+j], b = v[i+4+j];
                    v[i+j]   = _mm512_permutex2var_epi32(a, idx_a, b);
                    v[i+4+j] = _mm512_permutex2var_epi32(a, idx_b, b);
                }
            }
        }
        // Phase 4: 8-element interleave, pairs (0,8), (1,9), ..., (7,15)
        {
            ଅଚଳ __m512i idx_a = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
            ଅଚଳ __m512i idx_b = _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 8; i++) {
                __m512i a = v[i], b = v[i+8];
                v[i]   = _mm512_permutex2var_epi32(a, idx_a, b);
                v[i+8] = _mm512_permutex2var_epi32(a, idx_b, b);
            }
        }

        // store: v[g] holds 16 int32s for k-group col_order[g], rows 0..15
        ସ୍ଥିର ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା col_order[16] = {0, 8, 1, 9, 4, 12, 5, 13, 2, 10, 3, 11, 6, 14, 7, 15};
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା g = 0; g < 16; g++) {
            _mm512_store_si512((ଶୂନ୍ୟ *) (cbase + col_order[g] * row_stride), v[g]);
        }
    }
}
#ନଚେତଯଦି defined(__AVX2__)
// act-bias: pre-bias the activation in place (+128) so the MAC's maddubs(q1b, w) sees unsigned bytes;
// no transpose on AVX2 (natural layout). 16 rows x n_tiles*64 bytes, row r at base + r*row_stride
ଶୂନ୍ୟ tiled_repack_src1(tiled_tile_src1 * src1, ପୂର୍ଣ୍ଣସଂଖ୍ୟା row0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_k, ତାର୍କିକ bias) {
    ଯଦି (!bias) {
        ଫେରନ୍ତୁ;
    }
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row_stride = ଅଙ୍କ_k * TILED_TILE_K;
    uint8_t * base = (uint8_t *) src1->q + row0 * row_stride;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nbytes = row_stride; // full row width
    ଅଚଳ __m256i b128 = _mm256_set1_epi8((int8_t) 0x80);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା r = 0; r < TILED_MICRO; r++) {
        uint8_t * row = base + r * row_stride;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା off = 0; off < nbytes; off += 32) {
            ଅଚଳ __m256i v = _mm256_loadu_si256((ଅଚଳ __m256i *) (row + off));
            _mm256_storeu_si256((__m256i *) (row + off), _mm256_xor_si256(v, b128));
        }
    }
}
#ଅନ୍ୟଥା
ଶୂନ୍ୟ tiled_repack_src1(tiled_tile_src1 * src1, ପୂର୍ଣ୍ଣସଂଖ୍ୟା row0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_k, ତାର୍କିକ bias) {
    GGML_UNUSED(src1); GGML_UNUSED(row0); GGML_UNUSED(ଅଙ୍କ_k); GGML_UNUSED(bias);
}
#endif

#ଯଦି defined(__AVX2__)
// sum 32 unsigned bytes to int32 (via maddubs -> int16 pairs -> hsum)
ସ୍ଥିର int32_t tiled_byte_sum_32(ଅଚଳ uint8_t * p) {
    ଅଚଳ __m256i dot = _mm256_maddubs_epi16(_mm256_loadu_si256((ଅଚଳ __m256i *) p), _mm256_set1_epi8(1));
    ଅଚଳ __m256i sum = _mm256_madd_epi16(_mm256_set1_epi16(1), dot); // 8 int32
    // reduce 8 lanes: lo+hi -> 4 lanes, hadd -> 2, hadd -> 1 (a single hadd pair only folds 4 lanes)
    ଅଚଳ __m128i t = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    ଅଚଳ __m128i h = _mm_hadd_epi32(t, t);
    ଫେରନ୍ତୁ _mm_cvtsi128_si32(_mm_hadd_epi32(h, h));
}
// sum 16 unsigned bytes to int32
ସ୍ଥିର int32_t tiled_byte_sum_16(ଅଚଳ uint8_t * p) {
    ଅଚଳ __m128i dot = _mm_maddubs_epi16(_mm_loadu_si128((ଅଚଳ __m128i *) p), _mm_set1_epi8(1));
    ଅଚଳ __m128i sum = _mm_madd_epi16(_mm_set1_epi16(1), dot); // 4 int32
    ଅଚଳ __m128i h = _mm_hadd_epi32(sum, sum);
    ଫେରନ୍ତୁ _mm_cvtsi128_si32(_mm_hadd_epi32(h, h));
}
#endif

// act-bias: precompute the per-weight-row correction corr = 128*sum_s scales[s]*w_bsum[s] where
// w_bsum[s] = sum of the debiased weight bytes in subblock s. Stored in mins[r][0] (unused for
// HAS_MIN = false, which is exactly the act-bias case). Reused across all act bands and weight groups.
// No-op off AVX2 (the act-bias MAC is AVX2-only) and when corr is false (iq4_xs / BIAS = 0).
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା SUBBLK>
ଶୂନ୍ୟ tiled_repack_src0(tiled_tile_src0 * tile, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_rows, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_k, ପୂର୍ଣ୍ଣସଂଖ୍ୟା BIAS, ତାର୍କିକ corr) {
#ଯଦି defined(__AVX2__)
    ଯଦି (!corr) {
        ଫେରନ୍ତୁ;
    }
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା NB = TILED_TILE_K / SUBBLK;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb_stride = NB * ଅଙ୍କ_k;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା slab = 0; slab < ଅଙ୍କ_k; slab++) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା r = 0; r < n_rows; r++) {
            ଅଚଳ uint8_t * q = &tile->q[r * (ଅଙ୍କ_k * TILED_TILE_K) + slab * TILED_TILE_K];
            ଅଚଳ int32_t * scales = &tile->scales[r * nb_stride + slab * NB];
            int32_t corr_val = 0;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = 0; s < NB; s++) {
                ଅଚଳ int32_t qsum = (SUBBLK == 32) ? tiled_byte_sum_32(&q[s * SUBBLK]) : tiled_byte_sum_16(&q[s * SUBBLK]);
                corr_val += 128 * scales[s] * (qsum - SUBBLK * BIAS);
            }
            tile->mins[r * nb_stride + slab * NB] = corr_val;
        }
    }
#ଅନ୍ୟଥା
    GGML_UNUSED(tile); GGML_UNUSED(n_rows); GGML_UNUSED(ଅଙ୍କ_k); GGML_UNUSED(BIAS); GGML_UNUSED(corr);
#endif
}

template ଶୂନ୍ୟ tiled_repack_src0<16>(tiled_tile_src0 * tile, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_rows, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_k, ପୂର୍ଣ୍ଣସଂଖ୍ୟା BIAS, ତାର୍କିକ corr);
template ଶୂନ୍ୟ tiled_repack_src0<32>(tiled_tile_src0 * tile, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_rows, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_k, ପୂର୍ଣ୍ଣସଂଖ୍ୟା BIAS, ତାର୍କିକ corr);


