#ਹਦਾਇਤ once

// Tiled matmul kernel API: tile structs, kernel definitions

// Currently only optimized for x86, new architectures should implement:
// tiled_run_microtile:  16x16 microkernel
// tiled_repack_src0: Optional repack/recalculation of src0, per macrotile
// tiled_repack_src1: Optional repack/recalculation of src1, per microtile-band
// bit unpacking routines: tiled_unpk_nib4, tiled_unpk_2bit, tiled_unpk_or
// LUT value expansion routines: tiled_lut8, tiled_unpk_sign32, tiled_unpk_tern8

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_CPP
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stddef.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

#ਜੇ defined(__AVX2__)
#ਅਨ੍ਤਰ੍ਭਾਵ <immintrin.h>
#ਜੇ_ਅੰਤ

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TILED_TILE_K    256 // one QK_K block
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TILED_TILE_ਪੰਕਤੀਆਂ 256 // max window rows, ragged at edges
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TILED_MICRO     16  // microtile edge (also the bsums code-sum granularity)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TILED_WS_ਥਾਂ   (512 * 1024) // per-thread workspace slot, a clean 512KB (multiple of 64B)

// src0 tile: weight side, shared by all formats.
// scales/mins are sized for the max subblock count (SUBBLK=16);
// SUBBLK=32 formats index at stride 8 and leave the slack unused.
ਬਣਤਰ tiled_tile_src0 {
    ਸਥਿਰ constexpr ਪੂਰਨ NB_MAX = TILED_TILE_K / 16; // max subblocks per 256-elem block

    alignas(64) uint8_t q[TILED_TILE_ਪੰਕਤੀਆਂ * TILED_TILE_K]; // unsigned quants, widened to uint8
    ਦਸ਼ਮਲਵ    d[TILED_TILE_ਪੰਕਤੀਆਂ];  // One d from each input block, widened to f32
    ਦਸ਼ਮਲਵ    dmin[TILED_TILE_ਪੰਕਤੀਆਂ]; // dmin from each input block (if applicable), widened to F32
    int32_t   scales[TILED_TILE_ਪੰਕਤੀਆਂ * NB_MAX];  // per-subblock scale, stored as int32_t
    int32_t   mins[TILED_TILE_ਪੰਕਤੀਆਂ * NB_MAX];    // per-subblock min, used when HAS_MIN
};

// src1 tile: built from q8_K (wdata)
ਬਣਤਰ tiled_tile_src1 {
    // q8 codes, one byte per element. Note for VNNI these are reshaped + transposed to be suitable for dpbusd.
    alignas(64) int8_t  q[TILED_TILE_ਪੰਕਤੀਆਂ * TILED_TILE_K];
    // per-16 code sums from q8_k (int16), widened to int32 so the kernels load them directly, no per-use cvt
    alignas(64) int32_t bsums[(TILED_TILE_K / 16) * TILED_TILE_ਪੰਕਤੀਆਂ];
    // f32 (not f16): q8_k stores fp16, the unpack converts once
    ਦਸ਼ਮਲਵ       d[TILED_TILE_ਪੰਕਤੀਆਂ];
};

// per-thread workspace: all tiled state lives here, allocated in wdata (one slot per thread)
ਬਣਤਰ tiled_ws {
    tiled_tile_src0 src0;
    tiled_tile_src1 src1;
    alignas(64) ਦਸ਼ਮਲਵ acc[TILED_TILE_ਪੰਕਤੀਆਂ * TILED_TILE_ਪੰਕਤੀਆਂ];
};

static_assert(ਆਕਾਰ(tiled_ws) <= TILED_WS_ਥਾਂ, "tiled workspace exceeds the 512KB per-thread slot");
// the slot base is 64B-aligned and TILED_WS_SLOT is a multiple of 64B, so each per-thread slot
// is 64B-aligned; these pin the tile fields and the acc buffer at aligned offsets within a slot
static_assert(offsetof(tiled_ws, src0) % 64 == 0, "src0 not 64B-aligned in the workspace");
static_assert(offsetof(tiled_ws, src1) % 64 == 0, "src1 not 64B-aligned in the workspace");
static_assert(offsetof(tiled_ws, acc)  % 64 == 0, "acc not 64B-aligned in the workspace");

// unpack primitives for reading quants, defined as inline here to keep arch-specific code in kernel.h/.cpp
// If this section gets too hairy later, we can break up into separate includes.
#ਜੇ defined(__AVX2__)
// packed 4-bit codes -> low nibbles (lo) + high nibbles (hi)
ਸਤਰਅੰਦਰ ਖਾਲੀ tiled_unpk_nib4(ਅਚਲ uint8_t * src, uint8_t * lo, uint8_t * hi) {
    ਅਚਲ __m256i v = _mm256_loadu_si256((ਅਚਲ __m256i *) src);
    // mask before the lane shift so bits do not cross byte boundaries
    _mm256_storeu_si256((__m256i *) lo, _mm256_and_si256(v, _mm256_set1_epi8(0x0F)));
    _mm256_storeu_si256((__m256i *) hi, _mm256_srli_epi32(_mm256_and_si256(v, _mm256_set1_epi8((int8_t) 0xF0)), 4));
}
// 2-bit values at bit offset S
ਸਾਂਚਾ <ਪੂਰਨ S> ਸਤਰਅੰਦਰ ਖਾਲੀ tiled_unpk_2bit(ਅਚਲ uint8_t * src, uint8_t * dst) {
    _mm256_storeu_si256((__m256i *) dst, _mm256_and_si256(
        _mm256_srli_epi32(_mm256_loadu_si256((ਅਚਲ __m256i *) src), S), _mm256_set1_epi8(0x03)));
}
// OR the M-bit value at bit offset S of src into bit offset D of dst
ਸਾਂਚਾ <ਪੂਰਨ S, ਪੂਰਨ D, ਪੂਰਨ M>
ਸਤਰਅੰਦਰ ਖਾਲੀ tiled_unpk_or(uint8_t * dst, ਅਚਲ uint8_t * src) {
    ਅਚਲ __m256i v = _mm256_slli_epi32(_mm256_and_si256(
        _mm256_srli_epi32(_mm256_loadu_si256((ਅਚਲ __m256i *) src), S), _mm256_set1_epi8((uint8_t) M)), D);
    _mm256_storeu_si256((__m256i *) dst, _mm256_or_si256(_mm256_loadu_si256((ਅਚਲ __m256i *) dst), v));
}


// Unpacking kernels for IQ quants

// LUT value expansion for the LUT-based formats (iq4_xs, iq grids): the bit unpackers
// above give the indices, these expand 8/16 of them to widened codes in one pass
// 16-entry byte LUT: dst[j] = lut[src[j]] (16 bytes)
ਸਤਰਅੰਦਰ ਖਾਲੀ tiled_lut8(ਅਚਲ uint8_t * lut, ਅਚਲ uint8_t * src, uint8_t * dst) {
    _mm_storeu_si128((__m128i *) dst, _mm_shuffle_epi8(_mm_loadu_si128((ਅਚਲ __m128i *) lut),
                                                       _mm_loadu_si128((ਅਚਲ __m128i *) src)));
}


// 32 grid magnitudes (4 x 64-bit groups g0..g3, 8 values each) + 4 sign bytes
// (byte l signs values 8*l .. 8*l+7) -> codes stored as (value + 128)
ਸਤਰਅੰਦਰ ਖਾਲੀ tiled_unpk_sign32(uint64_t g0, uint64_t g1, uint64_t g2, uint64_t g3,
                              ਅਚਲ uint8_t signs[4], uint8_t * dst32) {
    ਅਚਲ __m256i v  = mm256_ਸੈੱਟ_epi64x((int64_t) g3, (int64_t) g2, (int64_t) g1, (int64_t) g0);
    ਅਚਲ __m256i sv = _mm256_shuffle_epi8(_mm256_set1_epi32((int32_t) (signs[0] | signs[1] << 8 | signs[2] << 16 | signs[3] << 24)),
                                           _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                                            2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
    ਅਚਲ __m256i sel  = _mm256_set1_epi64x((int64_t) 0x8040201008040201ULL);
    // 0xFF in every lane whose sign bit is set; GFNI does the and+compare in one instruction
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __GFNI__
    ਅਚਲ __m256i ਨਕਾਬ = _mm256_gf2p8affine_epi64_epi8(sel, sv, 0);
#ਨਹੀਂਤਾਂ
    ਅਚਲ __m256i ਨਕਾਬ = _mm256_cmpeq_epi8(_mm256_and_si256(sv, sel), sel);
#ਜੇ_ਅੰਤ
    // v ^ mask - mask negates the signed lanes (v elsewhere); + 128 gives the biased code
    ਅਚਲ __m256i sgn  = _mm256_sub_epi8(_mm256_xor_si256(v, ਨਕਾਬ), ਨਕਾਬ);
    _mm256_storeu_si256((__m256i *) dst32, mm256_ਜੋੜੋ_epi8(sgn, _mm256_set1_epi8((int8_t) 128)));
}
// 8 ternary grid bytes (0 = 0, 1 = +1, 0xFF = -1): dst[j] = 128 + delta + 8 * (int8_t) src[j]
ਸਤਰਅੰਦਰ ਖਾਲੀ tiled_unpk_tern8(ਅਚਲ uint8_t * src, int8_t delta, uint8_t * dst) {
    ਅਚਲ __m128i v = _mm_cvtepi8_epi16(_mm_loadl_epi64((ਅਚਲ __m128i *) src));
    ਅਚਲ __m128i p = mm_ਜੋੜੋ_epi16(_mm_slli_epi16(v, 3), _mm_set1_epi16(128 + (ਪੂਰਨ) delta));
    _mm_storel_epi64((__m128i *) dst, _mm_packus_epi16(p, _mm_setzero_si128()));
}
#ਨਹੀਂਤਾਂ

// Scalar definitions for unpackers.


ਸਤਰਅੰਦਰ ਖਾਲੀ tiled_unpk_nib4(ਅਚਲ uint8_t * src, uint8_t * lo, uint8_t * hi) {
    ਲਈ (ਪੂਰਨ l = 0; l < 32; l++) { lo[l] = (uint8_t) (src[l] & 0xF); hi[l] = (uint8_t) (src[l] >> 4); }
}
ਸਾਂਚਾ <ਪੂਰਨ S>
ਸਤਰਅੰਦਰ ਖਾਲੀ tiled_unpk_2bit(ਅਚਲ uint8_t * src, uint8_t * dst) {
    ਲਈ (ਪੂਰਨ l = 0; l < 32; l++) { dst[l] = (uint8_t) ((src[l] >> S) & 3); }
}
ਸਾਂਚਾ <ਪੂਰਨ S, ਪੂਰਨ D, ਪੂਰਨ M>
ਸਤਰਅੰਦਰ ਖਾਲੀ tiled_unpk_or(uint8_t * dst, ਅਚਲ uint8_t * src) {
    ਲਈ (ਪੂਰਨ l = 0; l < 32; l++) { dst[l] = (uint8_t) (dst[l] | (((src[l] >> S) & M) << D)); }
}
ਸਤਰਅੰਦਰ ਖਾਲੀ tiled_lut8(ਅਚਲ uint8_t * lut, ਅਚਲ uint8_t * src, uint8_t * dst) {
    ਲਈ (ਪੂਰਨ j = 0; j < 16; j++) { dst[j] = lut[src[j]]; }
}
ਸਤਰਅੰਦਰ ਖਾਲੀ tiled_unpk_sign32(uint64_t g0, uint64_t g1, uint64_t g2, uint64_t g3,
                              ਅਚਲ uint8_t signs[4], uint8_t * dst32) {
    ਅਚਲ uint64_t g[4] = { g0, g1, g2, g3 };
    ਲਈ (ਪੂਰਨ l = 0; l < 4; l++) {
        ਅਚਲ uint8_t * v = (ਅਚਲ uint8_t *) &g[l];
        ਅਚਲ uint8_t s = signs[l];
        ਲਈ (ਪੂਰਨ j = 0; j < 8; j++) {
            dst32[8 * l + j] = (s & (1 << j)) ? (uint8_t) (128 - v[j]) : (uint8_t) (128 + v[j]);
        }
    }
}
// 8 ternary grid bytes (0 = 0, 1 = +1, 0xFF = -1): dst[j] = 128 + delta + 8 * (int8_t) src[j]
ਸਤਰਅੰਦਰ ਖਾਲੀ tiled_unpk_tern8(ਅਚਲ uint8_t * src, int8_t delta, uint8_t * dst) {
    ਲਈ (ਪੂਰਨ j = 0; j < 8; j++) { dst[j] = (uint8_t) (128 + (ਪੂਰਨ) delta + 8 * (int8_t) src[j]); }
}
#ਜੇ_ਅੰਤ

// Accumulate one 16x16 microtile (src0 rows [i0, i0+16), src1 cols [j0, j0+16))
// over one 256-K slab held in the tiles into a j-major float buffer
// (row width buf_stride): buf[i*buf_stride + j] += partial.
// SUBBLK/HAS_MIN/BIAS are the src0 format constants (see tiled_tile_src0).
// ACTBIAS (AVX2 only): the activation is pre-biased +128 by tiled_repack_src1,
// num_k = K-blocks per row: the tile holds num_k slabs at row stride num_k*256
// (Default case is num_k=1, 256x256 tiles, we go to longer num_k to improve memory bandwidth when num_rows is small)
ਸਾਂਚਾ <ਪੂਰਨ SUBBLK, bool HAS_MIN, ਪੂਰਨ BIAS, bool ACTBIAS>
ਖਾਲੀ tiled_ਲੜੀਵਾਰ_microtile(ਅਚਲ tiled_tile_src0 & src0, ਅਚਲ tiled_tile_src1 & src1,
                         ਪੂਰਨ i0, ਪੂਰਨ j0, ਪੂਰਨ ਅੰਕ_k, ਪੂਰਨ slab, ਦਸ਼ਮਲਵ * buf, ਪੂਰਨ buf_stride);

// Optional repack, if profitable for the kernel.
// Repacks one 16-row band of src1 codes, called by driver as we reach each 16-row band in outer loop
ਖਾਲੀ tiled_repack_src1(tiled_tile_src1 * src1, ਪੂਰਨ row0, ਪੂਰਨ ਅੰਕ_k, bool bias);

// Optional repack, if profitable for the kernel
// Repack the entire src0 panel in place, called by the driver immediately after dequant
ਸਾਂਚਾ <ਪੂਰਨ SUBBLK>
ਖਾਲੀ tiled_repack_src0(tiled_tile_src0 * tile, ਪੂਰਨ n_ਪੰਕਤੀਆਂ, ਪੂਰਨ ਅੰਕ_k, ਪੂਰਨ BIAS, bool corr);


