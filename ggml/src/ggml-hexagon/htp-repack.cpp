#ಅನ್ತರ್ಭಾವ "htp-repack.h"

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_IMPL_CPP
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-quants.h"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <initializer_list>
#ಅನ್ತರ್ಭಾವ <thread>

#ಆದರೆ defined(__ARM_NEON) || defined(__aarch64__)
#    include <arm_neon.h>
#ಆದರೆ_ಅಂತ್ಯ

// --- 4-bit block-32 nibble helpers (shared by Q4_0, IQ4_NL, Q4_1, MXFP4) ---

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ unpack_nibbles_32(uint8_t * dst, ಸ್ಥಿರಾಂಕ uint8_t * qs) {
    ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i = 0; i < 16; ++i) {
        dst[i + 0]  = qs[i] & 0x0F;
        dst[i + 16] = qs[i] >> 4;
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ pack_nibbles_32(uint8_t * qs, ಸ್ಥಿರಾಂಕ uint8_t * src) {
    ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i = 0; i < 16; ++i) {
        qs[i] = src[i + 0] | (src[i + 16] << 4);
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ transpose_nibbles_ಗಮ್ಯ_tile(uint8_t * tile_dst, ಸ್ಥಿರಾಂಕ uint8_t tile_quants[32][32]) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ cp = 0; cp < 16; cp++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < 32; ಅಡ್ಡಸಾಲು++) {
            tile_dst[cp * 32 + ಅಡ್ಡಸಾಲು] = (tile_quants[ಅಡ್ಡಸಾಲು][2 * cp + 1] << 4) | tile_quants[ಅಡ್ಡಸಾಲು][2 * cp];
        }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ transpose_tile_ಗಮ್ಯ_nibbles(uint8_t tile_quants[32][32], ಸ್ಥಿರಾಂಕ uint8_t * tile_src) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ cp = 0; cp < 16; cp++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < 32; ಅಡ್ಡಸಾಲು++) {
            ಸ್ಥಿರಾಂಕ uint8_t val = tile_src[cp * 32 + ಅಡ್ಡಸಾಲು];
            tile_quants[ಅಡ್ಡಸಾಲು][2 * cp + 0] = val & 0x0F;
            tile_quants[ಅಡ್ಡಸಾಲು][2 * cp + 1] = val >> 4;
        }
    }
}

#ಆದರೆ defined(__ARM_NEON) || defined(__aarch64__)
ಸ್ಥಿರ ಶೂನ್ಯ repack_neon_transpose8x8(ಸ್ಥಿರಾಂಕ uint8x8_t ಅಡ್ಡಸಾಲುಗಳು[8], uint8x8_t columns[8]) {
    ಸ್ಥಿರಾಂಕ uint8x8x2_t t0 = vtrn_u8(ಅಡ್ಡಸಾಲುಗಳು[0], ಅಡ್ಡಸಾಲುಗಳು[1]);
    ಸ್ಥಿರಾಂಕ uint8x8x2_t t1 = vtrn_u8(ಅಡ್ಡಸಾಲುಗಳು[2], ಅಡ್ಡಸಾಲುಗಳು[3]);
    ಸ್ಥಿರಾಂಕ uint8x8x2_t t2 = vtrn_u8(ಅಡ್ಡಸಾಲುಗಳು[4], ಅಡ್ಡಸಾಲುಗಳು[5]);
    ಸ್ಥಿರಾಂಕ uint8x8x2_t t3 = vtrn_u8(ಅಡ್ಡಸಾಲುಗಳು[6], ಅಡ್ಡಸಾಲುಗಳು[7]);
    ಸ್ಥಿರಾಂಕ uint16x4x2_t t4 = vtrn_u16(vreinterpret_u16_u8(t0.val[0]), vreinterpret_u16_u8(t1.val[0]));
    ಸ್ಥಿರಾಂಕ uint16x4x2_t t5 = vtrn_u16(vreinterpret_u16_u8(t0.val[1]), vreinterpret_u16_u8(t1.val[1]));
    ಸ್ಥಿರಾಂಕ uint16x4x2_t t6 = vtrn_u16(vreinterpret_u16_u8(t2.val[0]), vreinterpret_u16_u8(t3.val[0]));
    ಸ್ಥಿರಾಂಕ uint16x4x2_t t7 = vtrn_u16(vreinterpret_u16_u8(t2.val[1]), vreinterpret_u16_u8(t3.val[1]));
    ಸ್ಥಿರಾಂಕ uint32x2x2_t t8 = vtrn_u32(vreinterpret_u32_u16(t4.val[0]), vreinterpret_u32_u16(t6.val[0]));
    ಸ್ಥಿರಾಂಕ uint32x2x2_t t9 = vtrn_u32(vreinterpret_u32_u16(t4.val[1]), vreinterpret_u32_u16(t6.val[1]));
    ಸ್ಥಿರಾಂಕ uint32x2x2_t ta = vtrn_u32(vreinterpret_u32_u16(t5.val[0]), vreinterpret_u32_u16(t7.val[0]));
    ಸ್ಥಿರಾಂಕ uint32x2x2_t tb = vtrn_u32(vreinterpret_u32_u16(t5.val[1]), vreinterpret_u32_u16(t7.val[1]));
    columns[0] = vreinterpret_u8_u32(t8.val[0]);
    columns[1] = vreinterpret_u8_u32(ta.val[0]);
    columns[2] = vreinterpret_u8_u32(t9.val[0]);
    columns[3] = vreinterpret_u8_u32(tb.val[0]);
    columns[4] = vreinterpret_u8_u32(t8.val[1]);
    columns[5] = vreinterpret_u8_u32(ta.val[1]);
    columns[6] = vreinterpret_u8_u32(t9.val[1]);
    columns[7] = vreinterpret_u8_u32(tb.val[1]);
}

ಸ್ಥಿರ uint8x8_t repack_neon_pack_4bit_ಜೋಡಿಗಳು(uint8x8_t ಮೊದಲ, uint8x8_t second, ತಾರ್ಕಿಕ ಮೇಲಿನ) {
    ಸ್ಥಿರಾಂಕ uint8x8x2_t separated = vuzp_u8(ಮೊದಲ, second);
    ಸ್ಥಿರಾಂಕ uint8x8_t ಮುಖವಾಡ = vdup_n_u8(0x0F);
    uint8x8_t even = separated.val[0];
    uint8x8_t odd  = separated.val[1];

    ಆದರೆ (ಮೇಲಿನ) {
        even = vshr_n_u8(even, 4);
        odd  = vshr_n_u8(odd, 4);
    }

    even = vand_u8(even, ಮುಖವಾಡ);
    odd  = vand_u8(odd, ಮುಖವಾಡ);
    ಹಿಂತಿರುಗಿ vorr_u8(vshl_n_u8(odd, 4), even);
}

ಸ್ಥಿರ ಶೂನ್ಯ repack_neon_4bit_ಅಡ್ಡಸಾಲುಗಳು(ಸ್ಥಿರಾಂಕ uint8_t * qs[8], uint8_t * tile_dst, ಪೂರ್ಣಾಂಕ row0) {
    uint8x8_t ಕೆಳಗಿನ_ಅಡ್ಡಸಾಲುಗಳು[8];
    uint8x8_t ಮೇಲಿನ_ಅಡ್ಡಸಾಲುಗಳು[8];
    uint8x8_t columns[8];

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ rr = 0; rr < 8; ++rr) {
        ಸ್ಥಿರಾಂಕ uint8x16_t packed = vld1q_u8(qs[rr]);
        ಸ್ಥಿರಾಂಕ uint8x8_t ಮೊದಲ = vget_ಕೆಳಗಿನ_u8(packed);
        ಸ್ಥಿರಾಂಕ uint8x8_t second = vget_ಮೇಲಿನ_u8(packed);
        ಕೆಳಗಿನ_ಅಡ್ಡಸಾಲುಗಳು[rr]  = repack_neon_pack_4bit_ಜೋಡಿಗಳು(ಮೊದಲ, second, ಸುಳ್ಳು);
        ಮೇಲಿನ_ಅಡ್ಡಸಾಲುಗಳು[rr] = repack_neon_pack_4bit_ಜೋಡಿಗಳು(ಮೊದಲ, second, ನಿಜ);
    }

    repack_neon_transpose8x8(ಕೆಳಗಿನ_ಅಡ್ಡಸಾಲುಗಳು, columns);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + cp * 32 + row0, columns[cp]);
    }

    repack_neon_transpose8x8(ಮೇಲಿನ_ಅಡ್ಡಸಾಲುಗಳು, columns);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + (cp + 8) * 32 + row0, columns[cp]);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ repack_neon_k4_ಅಡ್ಡಸಾಲುಗಳು(ಸ್ಥಿರಾಂಕ uint8_t * qs[8], uint8_t * tile_dst, ಪೂರ್ಣಾಂಕ row0, ತಾರ್ಕಿಕ ಮೇಲಿನ) {
    uint8x8_t ಮೊದಲ_ಅಡ್ಡಸಾಲುಗಳು[8];
    uint8x8_t ಎರಡನೇ_ಅಡ್ಡಸಾಲುಗಳು[8];
    uint8x8_t columns[8];

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ rr = 0; rr < 8; ++rr) {
        ಮೊದಲ_ಅಡ್ಡಸಾಲುಗಳು[rr] = repack_neon_pack_4bit_ಜೋಡಿಗಳು(vld1_u8(qs[rr]), vld1_u8(qs[rr] + 8), ಮೇಲಿನ);
        ಎರಡನೇ_ಅಡ್ಡಸಾಲುಗಳು[rr] = repack_neon_pack_4bit_ಜೋಡಿಗಳು(vld1_u8(qs[rr] + 16), vld1_u8(qs[rr] + 24), ಮೇಲಿನ);
    }

    repack_neon_transpose8x8(ಮೊದಲ_ಅಡ್ಡಸಾಲುಗಳು, columns);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + cp * 32 + row0, columns[cp]);
    }

    repack_neon_transpose8x8(ಎರಡನೇ_ಅಡ್ಡಸಾಲುಗಳು, columns);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + (cp + 8) * 32 + row0, columns[cp]);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ repack_neon_q8_ಅಡ್ಡಸಾಲುಗಳು(ಸ್ಥಿರಾಂಕ uint8_t * qs[8], uint8_t * tile_dst, ಪೂರ್ಣಾಂಕ row0) {
    uint8x8_t ಅಡ್ಡಸಾಲುಗಳು[8];
    uint8x8_t columns[8];

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ cp = 0; cp < 16; cp += 4) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ rr = 0; rr < 8; ++rr) {
            ಅಡ್ಡಸಾಲುಗಳು[rr] = vld1_u8(qs[rr] + 2 * cp);
        }
        repack_neon_transpose8x8(ಅಡ್ಡಸಾಲುಗಳು, columns);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ pair = 0; pair < 4; ++pair) {
            ಸ್ಥಿರಾಂಕ uint8x8x2_t interleaved = vzip_u8(columns[2 * pair], columns[2 * pair + 1]);
            uint8_t * dst = tile_dst + (cp + pair) * 64 + 2 * row0;
            vst1_u8(dst, interleaved.val[0]);
            vst1_u8(dst + 8, interleaved.val[1]);
        }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ repack_store_u32(uint8_t * dst, uint32_t v) {
    ಸ್ಮರಣೆ_ನಕಲು(dst, &v, ಗಾತ್ರ(v));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint32x4_t repack_neon_transpose_2bit_4x4(uint32x4_t w) {
    uint32x4_t t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 6)), vdupq_n_u32(0x00CC00CCu));
    w = veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 6)));
    t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 12)), vdupq_n_u32(0x0000F0F0u));
    ಹಿಂತಿರುಗಿ veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 12)));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ repack_neon_q2_0_ಖಂಡ(ಸ್ಥಿರಾಂಕ uint8_t * qs, uint8_t * tile, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು) {
    ಸ್ಥಿರಾಂಕ uint32x4_t w = repack_neon_transpose_2bit_4x4(vreinterpretq_u32_u8(vld1q_u8(qs)));
    repack_store_u32(tile + 0 * 128 + 4 * ಅಡ್ಡಸಾಲು, vgetq_lane_u32(w, 0));
    repack_store_u32(tile + 1 * 128 + 4 * ಅಡ್ಡಸಾಲು, vgetq_lane_u32(w, 1));
    repack_store_u32(tile + 2 * 128 + 4 * ಅಡ್ಡಸಾಲು, vgetq_lane_u32(w, 2));
    repack_store_u32(tile + 3 * 128 + 4 * ಅಡ್ಡಸಾಲು, vgetq_lane_u32(w, 3));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ repack_neon_q1_0_ಖಂಡ(ಸ್ಥಿರಾಂಕ uint8_t * qs, uint8_t * tile, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು) {
    uint32x4_t w = vreinterpretq_u32_u8(vld1q_u8(qs));
    uint32x4_t t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 1)), vdupq_n_u32(0x22222222u));
    w = veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 1)));
    t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 3)), vdupq_n_u32(0x0A0A0A0Au));
    w = veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 3)));
    w = repack_neon_transpose_2bit_4x4(w);
    repack_store_u32(tile + 0 * 128 + 4 * ಅಡ್ಡಸಾಲು, vgetq_lane_u32(w, 0));
    repack_store_u32(tile + 1 * 128 + 4 * ಅಡ್ಡಸಾಲು, vgetq_lane_u32(w, 1));
    repack_store_u32(tile + 2 * 128 + 4 * ಅಡ್ಡಸಾಲು, vgetq_lane_u32(w, 2));
    repack_store_u32(tile + 3 * 128 + 4 * ಅಡ್ಡಸಾಲು, vgetq_lane_u32(w, 3));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ repack_neon_pack_2bit_plane(uint8x16_t v0, uint8x16_t v1, uint8_t * dst, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು) {
    ಸ್ಥಿರಾಂಕ uint32x4x2_t z   = vuzpq_u32(vreinterpretq_u32_u8(v0), vreinterpretq_u32_u8(v1));
    ಸ್ಥಿರಾಂಕ uint32x4_t   p01 = vorrq_u32(z.val[0], vshlq_n_u32(z.val[1], 2));
    ಸ್ಥಿರಾಂಕ uint32x2x2_t u   = vuzp_u32(vget_ಕೆಳಗಿನ_u32(p01), vget_ಮೇಲಿನ_u32(p01));
    ಸ್ಥಿರಾಂಕ uint32x2_t   ಹೊರಹೋಗು = vorr_u32(u.val[0], vshl_n_u32(u.val[1], 4));
    repack_store_u32(dst + 0 * 128 + ಅಡ್ಡಸಾಲು * 4, vget_lane_u32(ಹೊರಹೋಗು, 0));
    repack_store_u32(dst + 1 * 128 + ಅಡ್ಡಸಾಲು * 4, vget_lane_u32(ಹೊರಹೋಗು, 1));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ repack_neon_q2_3_k_low2(ಸ್ಥಿರಾಂಕ uint8_t * qs, ಪೂರ್ಣಾಂಕ kt_local, uint8_t * lo_pl, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು) {
    ಸ್ಥಿರಾಂಕ uint8_t *  qs_sub = qs + (kt_local / 4) * 32;
    ಸ್ಥಿರಾಂಕ int8x16_t  ಸರಿಸು  = vdupq_n_s8(-2 * (kt_local % 4));
    ಸ್ಥಿರಾಂಕ uint8x16_t ಮುಖವಾಡ   = vdupq_n_u8(0x03);
    ಸ್ಥಿರಾಂಕ uint8x16_t v0     = vandq_u8(vshlq_u8(vld1q_u8(qs_sub +  0), ಸರಿಸು), ಮುಖವಾಡ);
    ಸ್ಥಿರಾಂಕ uint8x16_t v1     = vandq_u8(vshlq_u8(vld1q_u8(qs_sub + 16), ಸರಿಸು), ಮುಖವಾಡ);
    repack_neon_pack_2bit_plane(v0, v1, lo_pl, ಅಡ್ಡಸಾಲು);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ repack_neon_q3_k_hmask(ಸ್ಥಿರಾಂಕ uint8_t * hmask, ಪೂರ್ಣಾಂಕ kt_local, uint8_t * neg_pl, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು) {
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint32_t ಮುಖವಾಡ_ಪದಗಳು[4] = { 0x01010101u, 0x02020202u, 0x04040404u, 0x08080808u };
    ಸ್ಥಿರಾಂಕ uint8x16_t m0   = vreinterpretq_u8_u32(vld1q_u32(ಮುಖವಾಡ_ಪದಗಳು));
    ಸ್ಥಿರಾಂಕ uint8x16_t m1   = vshlq_n_u8(m0, 4);
    ಸ್ಥಿರಾಂಕ uint8x16_t hbit = vdupq_n_u8((uint8_t) (1u << kt_local));
    ಸ್ಥಿರಾಂಕ uint8x16_t b0   = vbicq_u8(m0, vtstq_u8(vld1q_u8(hmask +  0), hbit));
    ಸ್ಥಿರಾಂಕ uint8x16_t b1   = vbicq_u8(m1, vtstq_u8(vld1q_u8(hmask + 16), hbit));
    ಸ್ಥಿರಾಂಕ uint32x4_t b01  = vreinterpretq_u32_u8(vorrq_u8(b0, b1));
    ಸ್ಥಿರಾಂಕ uint32x2_t f0   = vorr_u32(vget_ಕೆಳಗಿನ_u32(b01), vget_ಮೇಲಿನ_u32(b01));
    ಸ್ಥಿರಾಂಕ uint32x2_t f1   = vpadd_u32(f0, f0);
    repack_store_u32(neg_pl + ಅಡ್ಡಸಾಲು * 4, vget_lane_u32(f1, 0));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ repack_neon_q6_k(ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K * b, ಪೂರ್ಣಾಂಕ kt_local, uint8_t * lo_pl, uint8_t * hi_pl, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ        c      = kt_local / 4;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ        g_sb   = kt_local % 4;
    ಸ್ಥಿರಾಂಕ uint8_t *  ql     = b->ql + c * 64 + (g_sb & 1) * 32;
    ಸ್ಥಿರಾಂಕ uint8_t *  qh     = b->qh + c * 32;
    ಸ್ಥಿರಾಂಕ int8x16_t  lshift = vdupq_n_s8((g_sb >= 2) ? -4 : 0);
    ಸ್ಥಿರಾಂಕ int8x16_t  hshift = vdupq_n_s8(-2 * g_sb);
    ಸ್ಥಿರಾಂಕ uint8x16_t mask4  = vdupq_n_u8(0x0F);
    ಸ್ಥಿರಾಂಕ uint8x16_t mask2  = vdupq_n_u8(0x03);

    ಸ್ಥಿರಾಂಕ uint8x16_t lo0  = vandq_u8(vshlq_u8(vld1q_u8(ql +  0), lshift), mask4);
    ಸ್ಥಿರಾಂಕ uint8x16_t lo1  = vandq_u8(vshlq_u8(vld1q_u8(ql + 16), lshift), mask4);
    ಸ್ಥಿರಾಂಕ uint32x4x2_t lz = vuzpq_u32(vreinterpretq_u32_u8(lo0), vreinterpretq_u32_u8(lo1));
    ಸ್ಥಿರಾಂಕ uint32x4_t lout = vorrq_u32(lz.val[0], vshlq_n_u32(lz.val[1], 4));
    repack_store_u32(lo_pl + 0 * 128 + ಅಡ್ಡಸಾಲು * 4, vgetq_lane_u32(lout, 0));
    repack_store_u32(lo_pl + 1 * 128 + ಅಡ್ಡಸಾಲು * 4, vgetq_lane_u32(lout, 1));
    repack_store_u32(lo_pl + 2 * 128 + ಅಡ್ಡಸಾಲು * 4, vgetq_lane_u32(lout, 2));
    repack_store_u32(lo_pl + 3 * 128 + ಅಡ್ಡಸಾಲು * 4, vgetq_lane_u32(lout, 3));

    ಸ್ಥಿರಾಂಕ uint8x16_t hi0 = vandq_u8(vshlq_u8(vld1q_u8(qh +  0), hshift), mask2);
    ಸ್ಥಿರಾಂಕ uint8x16_t hi1 = vandq_u8(vshlq_u8(vld1q_u8(qh + 16), hshift), mask2);
    repack_neon_pack_2bit_plane(hi0, hi1, hi_pl, ಅಡ್ಡಸಾಲು);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ repack_neon_q5_k_qh(ಸ್ಥಿರಾಂಕ uint8_t * qh, uint8_t hbit, uint8_t * plane, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು) {
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint8_t ಮುಖವಾಡ_ಬೈಟುಗಳು[16] = { 1, 2, 1, 2, 1, 2, 1, 2, 4, 8, 4, 8, 4, 8, 4, 8 };
    ಸ್ಥಿರಾಂಕ uint8x16_t m0   = vld1q_u8(ಮುಖವಾಡ_ಬೈಟುಗಳು);
    ಸ್ಥಿರಾಂಕ uint8x16_t m1   = vshlq_n_u8(m0, 4);
    ಸ್ಥಿರಾಂಕ uint8x16_t vbit = vdupq_n_u8(hbit);
    ಸ್ಥಿರಾಂಕ uint8x16_t b0   = vandq_u8(vtstq_u8(vld1q_u8(qh +  0), vbit), m0);
    ಸ್ಥಿರಾಂಕ uint8x16_t b1   = vandq_u8(vtstq_u8(vld1q_u8(qh + 16), vbit), m1);
    ಸ್ಥಿರಾಂಕ uint8x16_t b01  = vorrq_u8(b0, b1);
    ಸ್ಥಿರಾಂಕ uint8x8_t  fold = vorr_u8(vget_ಕೆಳಗಿನ_u8(b01), vget_ಮೇಲಿನ_u8(b01));
    ಸ್ಥಿರಾಂಕ uint8x8_t  ಹೊರಹೋಗು  = vpadd_u8(fold, fold);
    plane[0 * 32 + ಅಡ್ಡಸಾಲು] = vget_lane_u8(ಹೊರಹೋಗು, 0);
    plane[1 * 32 + ಅಡ್ಡಸಾಲು] = vget_lane_u8(ಹೊರಹೋಗು, 1);
    plane[2 * 32 + ಅಡ್ಡಸಾಲು] = vget_lane_u8(ಹೊರಹೋಗು, 2);
    plane[3 * 32 + ಅಡ್ಡಸಾಲು] = vget_lane_u8(ಹೊರಹೋಗು, 3);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint16x8_t repack_neon_ತುಂಬು_f16x8(ಸ್ಥಿರಾಂಕ uint16_t * src) {
    ಹಿಂತಿರುಗಿ vld1q_u16(src);
}

#ಆದರೆ defined(__aarch64__) || defined(__ARM_FP16_FORMAT_IEEE)
ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint16x8_t repack_neon_ತುಂಬು_f16x8(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src) {
    ಸ್ಥಿರಾಂಕ float16x4_t lo = vcvt_f16_f32(vld1q_f32(src));
    ಸ್ಥಿರಾಂಕ float16x4_t hi = vcvt_f16_f32(vld1q_f32(src + 4));
    ಹಿಂತಿರುಗಿ vreinterpretq_u16_f16(vcombine_f16(lo, hi));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint16x8_t repack_neon_ತುಂಬು_f16x8(ಸ್ಥಿರಾಂಕ ggml_bf16_t * src) {
    ಸ್ಥಿರಾಂಕ uint16x8_t  bf   = vld1q_u16((ಸ್ಥಿರಾಂಕ uint16_t *) src);
    ಸ್ಥಿರಾಂಕ float32x4_t f_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_ಕೆಳಗಿನ_u16(bf), 16));
    ಸ್ಥಿರಾಂಕ float32x4_t f_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_ಮೇಲಿನ_u16(bf), 16));
    ಸ್ಥಿರಾಂಕ float16x4_t lo   = vcvt_f16_f32(f_lo);
    ಸ್ಥಿರಾಂಕ float16x4_t hi   = vcvt_f16_f32(f_hi);
    ಹಿಂತಿರುಗಿ vreinterpretq_u16_f16(vcombine_f16(lo, hi));
}
#ಇಲ್ಲದಿದ್ದರೆ
ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint16x8_t repack_neon_ತುಂಬು_f16x8(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src) {
    uint16_t tmp[8];
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 8; ++i) {
        tmp[i] = GGML_FP32_ಗಮ್ಯ_FP16(src[i]);
    }
    ಹಿಂತಿರುಗಿ vld1q_u16(tmp);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint16x8_t repack_neon_ತುಂಬು_f16x8(ಸ್ಥಿರಾಂಕ ggml_bf16_t * src) {
    uint16_t tmp[8];
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 8; ++i) {
        tmp[i] = GGML_FP32_ಗಮ್ಯ_FP16(GGML_BF16_ಗಮ್ಯ_FP32(src[i]));
    }
    ಹಿಂತಿರುಗಿ vld1q_u16(tmp);
}
#ಆದರೆ_ಅಂತ್ಯ

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು SrcT>
ಸ್ಥಿರ ಶೂನ್ಯ repack_neon_f16_tile(uint16_t * tile_dst, ಸ್ಥಿರಾಂಕ SrcT * src_ತುಂಡು, int64_t ಅಡ್ಡಸಾಲು_stride, ಪೂರ್ಣಾಂಕ ct, ಪೂರ್ಣಾಂಕ kt) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ row0 = 0; row0 < 32; row0 += 4) {
        ಸ್ಥಿರಾಂಕ SrcT * r0 = src_ತುಂಡು + (int64_t) (ct * 32 + row0 + 0) * ಅಡ್ಡಸಾಲು_stride + kt * 32;
        ಸ್ಥಿರಾಂಕ SrcT * r1 = src_ತುಂಡು + (int64_t) (ct * 32 + row0 + 1) * ಅಡ್ಡಸಾಲು_stride + kt * 32;
        ಸ್ಥಿರಾಂಕ SrcT * r2 = src_ತುಂಡು + (int64_t) (ct * 32 + row0 + 2) * ಅಡ್ಡಸಾಲು_stride + kt * 32;
        ಸ್ಥಿರಾಂಕ SrcT * r3 = src_ತುಂಡು + (int64_t) (ct * 32 + row0 + 3) * ಅಡ್ಡಸಾಲು_stride + kt * 32;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ cp = 0; cp < 16; cp += 4) {
            ಸ್ಥಿರಾಂಕ uint32x4_t v0 = vreinterpretq_u32_u16(repack_neon_ತುಂಬು_f16x8(r0 + 2 * cp));
            ಸ್ಥಿರಾಂಕ uint32x4_t v1 = vreinterpretq_u32_u16(repack_neon_ತುಂಬು_f16x8(r1 + 2 * cp));
            ಸ್ಥಿರಾಂಕ uint32x4_t v2 = vreinterpretq_u32_u16(repack_neon_ತುಂಬು_f16x8(r2 + 2 * cp));
            ಸ್ಥಿರಾಂಕ uint32x4_t v3 = vreinterpretq_u32_u16(repack_neon_ತುಂಬು_f16x8(r3 + 2 * cp));

            ಸ್ಥಿರಾಂಕ uint32x2x2_t t0 = vtrn_u32(vget_ಕೆಳಗಿನ_u32(v0),  vget_ಕೆಳಗಿನ_u32(v1));
            ಸ್ಥಿರಾಂಕ uint32x2x2_t t1 = vtrn_u32(vget_ಮೇಲಿನ_u32(v0), vget_ಮೇಲಿನ_u32(v1));
            ಸ್ಥಿರಾಂಕ uint32x2x2_t t2 = vtrn_u32(vget_ಕೆಳಗಿನ_u32(v2),  vget_ಕೆಳಗಿನ_u32(v3));
            ಸ್ಥಿರಾಂಕ uint32x2x2_t t3 = vtrn_u32(vget_ಮೇಲಿನ_u32(v2), vget_ಮೇಲಿನ_u32(v3));

            vst1q_u16(tile_dst + (cp + 0) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t0.val[0], t2.val[0])));
            vst1q_u16(tile_dst + (cp + 1) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t0.val[1], t2.val[1])));
            vst1q_u16(tile_dst + (cp + 2) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t1.val[0], t3.val[0])));
            vst1q_u16(tile_dst + (cp + 3) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t1.val[1], t3.val[1])));
        }
    }
}
#ಆದರೆ_ಅಂತ್ಯ

// --- Generic 32x32 tile grid walkers for block-32 and float types ---

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು Fn>
ಸ್ಥಿರ ಶೂನ್ಯ repack_parallel_for_ct(size_t ತುಂಡು_ಬೈಟುಗಳು, ಪೂರ್ಣಾಂಕ n_ಲಂಬಸಾಲು_tiles, Fn && fn) {
    ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು = 1;
    ಆದರೆ (ತುಂಡು_ಬೈಟುಗಳು > 128 * 1024) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ nt : { 4, 3, 2 }) {
            ಆದರೆ (n_ಲಂಬಸಾಲು_tiles % nt == 0) {
                n_ಎಳೆಗಳು = nt;
                ನಿಲ್ಲಿಸು;
            }
        }
    }

    ಆದರೆ (n_ಎಳೆಗಳು <= 1) {
        fn(0, n_ಲಂಬಸಾಲು_tiles);
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ chunk = n_ಲಂಬಸಾಲು_tiles / n_ಎಳೆಗಳು;
    std::thread workers[3];
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n_ಎಳೆಗಳು - 1; ++i) {
        workers[i] = std::thread([&fn, i, chunk]() {
            fn(i * chunk, (i + 1) * chunk);
        });
    }
    fn((n_ಎಳೆಗಳು - 1) * chunk, n_ಲಂಬಸಾಲು_tiles);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n_ಎಳೆಗಳು - 1; ++i) {
        workers[i].join();
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು SrcElemT, ಪೂರ್ಣಾಂಕ ELEMS_ಪ್ರತಿ_KTILE, ಬಗೆ_ಹೆಸರು PackTileFn>
ಸ್ಥಿರ ಶೂನ್ಯ repack_ಗಮ್ಯ_tiled_grid_32(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size,
                                    size_t tile_ಗಾತ್ರ, PackTileFn && pack_tile) {
    ಸ್ಥಿರಾಂಕ SrcElemT * src_ಮಾತೃಕೆ = (ಸ್ಥಿರಾಂಕ SrcElemT *) ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ int64_t ne0 = t->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne1 = t->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne2 = t->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne3 = t->ne[3];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ಲಂಬಸಾಲು_tiles = hex_ಸುತ್ತು_ಮೇಲೆ(ne1, 32) / 32;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_k_tiles   = hex_ಸುತ್ತು_ಮೇಲೆ(ne0, 32) / 32;
    ಸ್ಥಿರಾಂಕ size_t ಮಾತೃಕೆ_ಗಾತ್ರ = (size_t) n_ಲಂಬಸಾಲು_tiles * n_k_tiles * tile_ಗಾತ್ರ;

    ಸ್ಥಿರಾಂಕ size_t ತುಂಡು_ಗಾತ್ರ = (size_t) ne1 * ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(t->type, ne0);
    ಸ್ಥಿರಾಂಕ int64_t ಆರಂಭ_ತುಂಡು = ಸ್ಥಾನಾಂತರ / ತುಂಡು_ಗಾತ್ರ;
    int64_t ಕೊನೆ_ತುಂಡು = (ಸ್ಥಾನಾಂತರ + size + ತುಂಡು_ಗಾತ್ರ - 1) / ತುಂಡು_ಗಾತ್ರ;
    ಆದರೆ (ಕೊನೆ_ತುಂಡು > ne2 * ne3) {
        ಕೊನೆ_ತುಂಡು = ne2 * ne3;
    }

    ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_stride = (ne0 / (32 / ELEMS_ಪ್ರತಿ_KTILE));

    ಗಾಗಿ (int64_t ತುಂಡು_idx = ಆರಂಭ_ತುಂಡು; ತುಂಡು_idx < ಕೊನೆ_ತುಂಡು; ತುಂಡು_idx++) {
        ಸ್ಥಿರಾಂಕ SrcElemT * src_ತುಂಡು = src_ಮಾತೃಕೆ + (ತುಂಡು_idx - ಆರಂಭ_ತುಂಡು) * (ne1 * ಅಡ್ಡಸಾಲು_stride);
        uint8_t * ಮಾತೃಕೆ_dst = (uint8_t *) t->ದತ್ತಾಂಶ + ತುಂಡು_idx * ಮಾತೃಕೆ_ಗಾತ್ರ;

        repack_parallel_for_ct(ಮಾತೃಕೆ_ಗಾತ್ರ, n_ಲಂಬಸಾಲು_tiles, [&](ಪೂರ್ಣಾಂಕ ct0, ಪೂರ್ಣಾಂಕ ct1) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ct = ct0; ct < ct1; ct++) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = ಮಾತೃಕೆ_dst + ((size_t) ct * n_k_tiles + kt) * tile_ಗಾತ್ರ;
                    pack_tile(tile_dst, src_ತುಂಡು, ne0, ne1, ಅಡ್ಡಸಾಲು_stride, ct, kt);
                }
            }
        });
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು DstElemT, ಪೂರ್ಣಾಂಕ ELEMS_ಪ್ರತಿ_KTILE, ಬಗೆ_ಹೆಸರು UnpackTileFn>
ಸ್ಥಿರ ಶೂನ್ಯ unpack_from_tiled_grid_32(ಶೂನ್ಯ * ದತ್ತಾಂಶ, ಸ್ಥಿರಾಂಕ ggml_tensor * t, size_t ಸ್ಥಾನಾಂತರ, size_t size,
                                      size_t tile_ಗಾತ್ರ, UnpackTileFn && unpack_tile) {
    uint8_t * dst_ಬೈಟುಗಳು = (uint8_t *) ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ int64_t ne0 = t->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne1 = t->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne2 = t->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne3 = t->ne[3];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ಲಂಬಸಾಲು_tiles = hex_ಸುತ್ತು_ಮೇಲೆ(ne1, 32) / 32;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_k_tiles   = hex_ಸುತ್ತು_ಮೇಲೆ(ne0, 32) / 32;
    ಸ್ಥಿರಾಂಕ size_t ಮಾತೃಕೆ_ಗಾತ್ರ = (size_t) n_ಲಂಬಸಾಲು_tiles * n_k_tiles * tile_ಗಾತ್ರ;

    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು = ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(t->type, ne0);
    ಸ್ಥಿರಾಂಕ size_t ತುಂಡು_ಗಾತ್ರ     = (size_t) ne1 * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು;
    ಸ್ಥಿರಾಂಕ int64_t ಆರಂಭ_ತುಂಡು   = ಸ್ಥಾನಾಂತರ / ತುಂಡು_ಗಾತ್ರ;
    int64_t ಕೊನೆ_ತುಂಡು           = (ಸ್ಥಾನಾಂತರ + size + ತುಂಡು_ಗಾತ್ರ - 1) / ತುಂಡು_ಗಾತ್ರ;
    ಆದರೆ (ಕೊನೆ_ತುಂಡು > ne2 * ne3) {
        ಕೊನೆ_ತುಂಡು = ne2 * ne3;
    }

    ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_stride = (ne0 / (32 / ELEMS_ಪ್ರತಿ_KTILE));

    ಗಾಗಿ (int64_t ತುಂಡು_idx = ಆರಂಭ_ತುಂಡು; ತುಂಡು_idx < ಕೊನೆ_ತುಂಡು; ತುಂಡು_idx++) {
        ಸ್ಥಿರಾಂಕ size_t cur_ಆರಂಭ_byte     = (std::max)(ಸ್ಥಾನಾಂತರ, (size_t) ತುಂಡು_idx * ತುಂಡು_ಗಾತ್ರ);
        ಸ್ಥಿರಾಂಕ size_t cur_ಕೊನೆ_byte       = (std::min)(ಸ್ಥಾನಾಂತರ + size, (size_t) (ತುಂಡು_idx + 1) * ತುಂಡು_ಗಾತ್ರ);
        ಸ್ಥಿರಾಂಕ size_t ತುಂಡು_ಸ್ಥಾನಾಂತರ_ಆರಂಭ = cur_ಆರಂಭ_byte - (size_t) ತುಂಡು_idx * ತುಂಡು_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ size_t ತುಂಡು_ಸ್ಥಾನಾಂತರ_ಕೊನೆ   = cur_ಕೊನೆ_byte - (size_t) ತುಂಡು_idx * ತುಂಡು_ಗಾತ್ರ;

        ಸ್ಥಿರಾಂಕ int64_t ಆರಂಭ_ಅಡ್ಡಸಾಲು = ತುಂಡು_ಸ್ಥಾನಾಂತರ_ಆರಂಭ / ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು;
        ಸ್ಥಿರಾಂಕ int64_t ಕೊನೆ_ಅಡ್ಡಸಾಲು   = (std::min)((int64_t) ((ತುಂಡು_ಸ್ಥಾನಾಂತರ_ಕೊನೆ + ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು - 1) / ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು), ne1);

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಆರಂಭ_ct = (ಪೂರ್ಣಾಂಕ) (ಆರಂಭ_ಅಡ್ಡಸಾಲು / 32);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಕೊನೆ_ct   = (std::min)((ಪೂರ್ಣಾಂಕ) ((ಕೊನೆ_ಅಡ್ಡಸಾಲು + 31) / 32), n_ಲಂಬಸಾಲು_tiles);

        DstElemT * dst_ತುಂಡು = (DstElemT *) (dst_ಬೈಟುಗಳು + (cur_ಆರಂಭ_byte - ಸ್ಥಾನಾಂತರ));
        ಸ್ಥಿರಾಂಕ uint8_t * ಮಾತೃಕೆ_src = (ಸ್ಥಿರಾಂಕ uint8_t *) t->ದತ್ತಾಂಶ + ತುಂಡು_idx * ಮಾತೃಕೆ_ಗಾತ್ರ;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ct = ಆರಂಭ_ct; ct < ಕೊನೆ_ct; ct++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ kt = 0; kt < n_k_tiles; kt++) {
                ಸ್ಥಿರಾಂಕ uint8_t * tile_src = ಮಾತೃಕೆ_src + ((size_t) ct * n_k_tiles + kt) * tile_ಗಾತ್ರ;
                unpack_tile(dst_ತುಂಡು, tile_src, ne0, ಆರಂಭ_ಅಡ್ಡಸಾಲು, ಕೊನೆ_ಅಡ್ಡಸಾಲು, ಅಡ್ಡಸಾಲು_stride, ct, kt);
            }
        }
    }
}

// --- Block-32 Quantized Types (Q4_0, IQ4_NL, Q4_1, Q8_0, MXFP4) ---

ಸ್ಥಿರ ಶೂನ್ಯ repack_q4_0_tiled(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    repack_ಗಮ್ಯ_tiled_grid_32<ಖಂಡ_q4_0, 1>(t, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, HTP_TENSOR_TILE_ಗಾತ್ರ_Q4_0,
        [](uint8_t * tile_dst, ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * src_ತುಂಡು, int64_t ne0, int64_t ne1, int64_t ಅಡ್ಡಸಾಲು_stride, ಪೂರ್ಣಾಂಕ ct, ಪೂರ್ಣಾಂಕ kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
#ಆದರೆ defined(__ARM_NEON) || defined(__aarch64__)
            ಆದರೆ (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ row0 = 0; row0 < 32; row0 += 8) {
                    ಸ್ಥಿರಾಂಕ uint8_t * qs[8];
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_ತುಂಡು[(ct * 32 + row0 + rr) * ಅಡ್ಡಸಾಲು_stride + kt].qs;
                    }
                    repack_neon_4bit_ಅಡ್ಡಸಾಲುಗಳು(qs, tile_dst, row0);
                }

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < 32; ++ಅಡ್ಡಸಾಲು) {
                    scale_dst[ಅಡ್ಡಸಾಲು] = src_ತುಂಡು[(ct * 32 + ಅಡ್ಡಸಾಲು) * ಅಡ್ಡಸಾಲು_stride + kt].d;
                }
                ಹಿಂತಿರುಗಿ;
            }
#ಆದರೆ_ಅಂತ್ಯ
            uint8_t tile_quants[32][32];
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < 32; ಅಡ್ಡಸಾಲು++) {
                ಸ್ಥಿರಾಂಕ int64_t r = ct * 32 + ಅಡ್ಡಸಾಲು;
                ಆದರೆ (r < ne1 && kt < ne0 / 32) {
                    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 & b = src_ತುಂಡು[r * ಅಡ್ಡಸಾಲು_stride + kt];
                    unpack_nibbles_32(tile_quants[ಅಡ್ಡಸಾಲು], b.qs);
                    scale_dst[ಅಡ್ಡಸಾಲು] = b.d;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    memset(tile_quants[ಅಡ್ಡಸಾಲು], 8, 32);
                    scale_dst[ಅಡ್ಡಸಾಲು] = 0;
                }
            }
            transpose_nibbles_ಗಮ್ಯ_tile(tile_dst, tile_quants);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ repack_tiled_q4_0(ಶೂನ್ಯ * ದತ್ತಾಂಶ, ಸ್ಥಿರಾಂಕ ggml_tensor * t, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    unpack_from_tiled_grid_32<ಖಂಡ_q4_0, 1>(ದತ್ತಾಂಶ, t, ಸ್ಥಾನಾಂತರ, size, HTP_TENSOR_TILE_ಗಾತ್ರ_Q4_0,
        [](ಖಂಡ_q4_0 * dst_ತುಂಡು, ಸ್ಥಿರಾಂಕ uint8_t * tile_src, int64_t ne0, int64_t ಆರಂಭ_ಅಡ್ಡಸಾಲು, int64_t ಕೊನೆ_ಅಡ್ಡಸಾಲು, int64_t ಅಡ್ಡಸಾಲು_stride, ಪೂರ್ಣಾಂಕ ct, ಪೂರ್ಣಾಂಕ kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_ಗಮ್ಯ_nibbles(tile_quants, tile_src);
            ಸ್ಥಿರಾಂಕ ggml_half * scale_src = (ಸ್ಥಿರಾಂಕ ggml_half *) (tile_src + 512);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < 32; ಅಡ್ಡಸಾಲು++) {
                ಸ್ಥಿರಾಂಕ int64_t r = ct * 32 + ಅಡ್ಡಸಾಲು;
                ಆದರೆ (r >= ಆರಂಭ_ಅಡ್ಡಸಾಲು && r < ಕೊನೆ_ಅಡ್ಡಸಾಲು && kt < ne0 / 32) {
                    ಖಂಡ_q4_0 & b = dst_ತುಂಡು[(r - ಆರಂಭ_ಅಡ್ಡಸಾಲು) * ಅಡ್ಡಸಾಲು_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[ಅಡ್ಡಸಾಲು]);
                    b.d = scale_src[ಅಡ್ಡಸಾಲು];
                }
            }
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ repack_q4_1_tiled(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    repack_ಗಮ್ಯ_tiled_grid_32<ಖಂಡ_q4_1, 1>(t, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, HTP_TENSOR_TILE_ಗಾತ್ರ_Q4_1,
        [](uint8_t * tile_dst, ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 * src_ತುಂಡು, int64_t ne0, int64_t ne1, int64_t ಅಡ್ಡಸಾಲು_stride, ಪೂರ್ಣಾಂಕ ct, ಪೂರ್ಣಾಂಕ kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
#ಆದರೆ defined(__ARM_NEON) || defined(__aarch64__)
            ಆದರೆ (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ row0 = 0; row0 < 32; row0 += 8) {
                    ಸ್ಥಿರಾಂಕ uint8_t * qs[8];
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_ತುಂಡು[(ct * 32 + row0 + rr) * ಅಡ್ಡಸಾಲು_stride + kt].qs;
                    }
                    repack_neon_4bit_ಅಡ್ಡಸಾಲುಗಳು(qs, tile_dst, row0);
                }

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < 32; ++ಅಡ್ಡಸಾಲು) {
                    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 & src_ಖಂಡ = src_ತುಂಡು[(ct * 32 + ಅಡ್ಡಸಾಲು) * ಅಡ್ಡಸಾಲು_stride + kt];
                    scale_dst[2 * ಅಡ್ಡಸಾಲು + 0] = src_ಖಂಡ.d;
                    scale_dst[2 * ಅಡ್ಡಸಾಲು + 1] = src_ಖಂಡ.m;
                }
                ಹಿಂತಿರುಗಿ;
            }
#ಆದರೆ_ಅಂತ್ಯ
            uint8_t tile_quants[32][32];
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < 32; ಅಡ್ಡಸಾಲು++) {
                ಸ್ಥಿರಾಂಕ int64_t r = ct * 32 + ಅಡ್ಡಸಾಲು;
                ಆದರೆ (r < ne1 && kt < ne0 / 32) {
                    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 & b = src_ತುಂಡು[r * ಅಡ್ಡಸಾಲು_stride + kt];
                    unpack_nibbles_32(tile_quants[ಅಡ್ಡಸಾಲು], b.qs);
                    scale_dst[2 * ಅಡ್ಡಸಾಲು + 0] = b.d;
                    scale_dst[2 * ಅಡ್ಡಸಾಲು + 1] = b.m;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    memset(tile_quants[ಅಡ್ಡಸಾಲು], 0, 32);
                    scale_dst[2 * ಅಡ್ಡಸಾಲು + 0] = 0;
                    scale_dst[2 * ಅಡ್ಡಸಾಲು + 1] = 0;
                }
            }
            transpose_nibbles_ಗಮ್ಯ_tile(tile_dst, tile_quants);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ repack_tiled_q4_1(ಶೂನ್ಯ * ದತ್ತಾಂಶ, ಸ್ಥಿರಾಂಕ ggml_tensor * t, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    unpack_from_tiled_grid_32<ಖಂಡ_q4_1, 1>(ದತ್ತಾಂಶ, t, ಸ್ಥಾನಾಂತರ, size, HTP_TENSOR_TILE_ಗಾತ್ರ_Q4_1,
        [](ಖಂಡ_q4_1 * dst_ತುಂಡು, ಸ್ಥಿರಾಂಕ uint8_t * tile_src, int64_t ne0, int64_t ಆರಂಭ_ಅಡ್ಡಸಾಲು, int64_t ಕೊನೆ_ಅಡ್ಡಸಾಲು, int64_t ಅಡ್ಡಸಾಲು_stride, ಪೂರ್ಣಾಂಕ ct, ಪೂರ್ಣಾಂಕ kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_ಗಮ್ಯ_nibbles(tile_quants, tile_src);
            ಸ್ಥಿರಾಂಕ ggml_half * scale_src = (ಸ್ಥಿರಾಂಕ ggml_half *) (tile_src + 512);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < 32; ಅಡ್ಡಸಾಲು++) {
                ಸ್ಥಿರಾಂಕ int64_t r = ct * 32 + ಅಡ್ಡಸಾಲು;
                ಆದರೆ (r >= ಆರಂಭ_ಅಡ್ಡಸಾಲು && r < ಕೊನೆ_ಅಡ್ಡಸಾಲು && kt < ne0 / 32) {
                    ಖಂಡ_q4_1 & b = dst_ತುಂಡು[(r - ಆರಂಭ_ಅಡ್ಡಸಾಲು) * ಅಡ್ಡಸಾಲು_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[ಅಡ್ಡಸಾಲು]);
                    b.d = scale_src[2 * ಅಡ್ಡಸಾಲು + 0];
                    b.m = scale_src[2 * ಅಡ್ಡಸಾಲು + 1];
                }
            }
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ repack_mxfp4_tiled(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    repack_ಗಮ್ಯ_tiled_grid_32<ಖಂಡ_mxfp4, 1>(t, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, HTP_TENSOR_TILE_ಗಾತ್ರ_MXFP4,
        [](uint8_t * tile_dst, ಸ್ಥಿರಾಂಕ ಖಂಡ_mxfp4 * src_ತುಂಡು, int64_t ne0, int64_t ne1, int64_t ಅಡ್ಡಸಾಲು_stride, ಪೂರ್ಣಾಂಕ ct, ಪೂರ್ಣಾಂಕ kt) {
            uint8_t * scale_dst = tile_dst + 512;
#ಆದರೆ defined(__ARM_NEON) || defined(__aarch64__)
            ಆದರೆ (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ row0 = 0; row0 < 32; row0 += 8) {
                    ಸ್ಥಿರಾಂಕ uint8_t * qs[8];
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_ತುಂಡು[(ct * 32 + row0 + rr) * ಅಡ್ಡಸಾಲು_stride + kt].qs;
                    }
                    repack_neon_4bit_ಅಡ್ಡಸಾಲುಗಳು(qs, tile_dst, row0);
                }

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < 32; ++ಅಡ್ಡಸಾಲು) {
                    scale_dst[ಅಡ್ಡಸಾಲು] = src_ತುಂಡು[(ct * 32 + ಅಡ್ಡಸಾಲು) * ಅಡ್ಡಸಾಲು_stride + kt].e;
                }
                ಹಿಂತಿರುಗಿ;
            }
#ಆದರೆ_ಅಂತ್ಯ
            uint8_t tile_quants[32][32];
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < 32; ಅಡ್ಡಸಾಲು++) {
                ಸ್ಥಿರಾಂಕ int64_t r = ct * 32 + ಅಡ್ಡಸಾಲು;
                ಆದರೆ (r < ne1 && kt < ne0 / 32) {
                    ಸ್ಥಿರಾಂಕ ಖಂಡ_mxfp4 & b = src_ತುಂಡು[r * ಅಡ್ಡಸಾಲು_stride + kt];
                    unpack_nibbles_32(tile_quants[ಅಡ್ಡಸಾಲು], b.qs);
                    scale_dst[ಅಡ್ಡಸಾಲು] = b.e;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    memset(tile_quants[ಅಡ್ಡಸಾಲು], 0, 32);
                    scale_dst[ಅಡ್ಡಸಾಲು] = 0;
                }
            }
            transpose_nibbles_ಗಮ್ಯ_tile(tile_dst, tile_quants);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ repack_tiled_mxfp4(ಶೂನ್ಯ * ದತ್ತಾಂಶ, ಸ್ಥಿರಾಂಕ ggml_tensor * t, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    unpack_from_tiled_grid_32<ಖಂಡ_mxfp4, 1>(ದತ್ತಾಂಶ, t, ಸ್ಥಾನಾಂತರ, size, HTP_TENSOR_TILE_ಗಾತ್ರ_MXFP4,
        [](ಖಂಡ_mxfp4 * dst_ತುಂಡು, ಸ್ಥಿರಾಂಕ uint8_t * tile_src, int64_t ne0, int64_t ಆರಂಭ_ಅಡ್ಡಸಾಲು, int64_t ಕೊನೆ_ಅಡ್ಡಸಾಲು, int64_t ಅಡ್ಡಸಾಲು_stride, ಪೂರ್ಣಾಂಕ ct, ಪೂರ್ಣಾಂಕ kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_ಗಮ್ಯ_nibbles(tile_quants, tile_src);
            ಸ್ಥಿರಾಂಕ uint8_t * scale_src = tile_src + 512;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < 32; ಅಡ್ಡಸಾಲು++) {
                ಸ್ಥಿರಾಂಕ int64_t r = ct * 32 + ಅಡ್ಡಸಾಲು;
                ಆದರೆ (r >= ಆರಂಭ_ಅಡ್ಡಸಾಲು && r < ಕೊನೆ_ಅಡ್ಡಸಾಲು && kt < ne0 / 32) {
                    ಖಂಡ_mxfp4 & b = dst_ತುಂಡು[(r - ಆರಂಭ_ಅಡ್ಡಸಾಲು) * ಅಡ್ಡಸಾಲು_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[ಅಡ್ಡಸಾಲು]);
                    b.e = scale_src[ಅಡ್ಡಸಾಲು];
                }
            }
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ repack_q8_0_tiled(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    repack_ಗಮ್ಯ_tiled_grid_32<ಖಂಡ_q8_0, 1>(t, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, HTP_TENSOR_TILE_ಗಾತ್ರ_Q8_0,
        [](uint8_t * tile_dst, ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * src_ತುಂಡು, int64_t ne0, int64_t ne1, int64_t ಅಡ್ಡಸಾಲು_stride, ಪೂರ್ಣಾಂಕ ct, ಪೂರ್ಣಾಂಕ kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 1024);
#ಆದರೆ defined(__ARM_NEON) || defined(__aarch64__)
            ಆದರೆ (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ row0 = 0; row0 < 32; row0 += 8) {
                    ಸ್ಥಿರಾಂಕ uint8_t * qs[8];
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ rr = 0; rr < 8; ++rr) {
                        qs[rr] = (ಸ್ಥಿರಾಂಕ uint8_t *) src_ತುಂಡು[(ct * 32 + row0 + rr) * ಅಡ್ಡಸಾಲು_stride + kt].qs;
                    }
                    repack_neon_q8_ಅಡ್ಡಸಾಲುಗಳು(qs, tile_dst, row0);
                }

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < 32; ++ಅಡ್ಡಸಾಲು) {
                    scale_dst[ಅಡ್ಡಸಾಲು] = src_ತುಂಡು[(ct * 32 + ಅಡ್ಡಸಾಲು) * ಅಡ್ಡಸಾಲು_stride + kt].d;
                }
                ಹಿಂತಿರುಗಿ;
            }
#ಆದರೆ_ಅಂತ್ಯ
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ cp = 0; cp < 16; cp++) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ col0 = cp * 2;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ col1 = col0 + 1;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < 32; ಅಡ್ಡಸಾಲು++) {
                    ಸ್ಥಿರಾಂಕ int64_t r = ct * 32 + ಅಡ್ಡಸಾಲು;
                    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * b = (r < ne1 && kt < ne0 / 32) ? &src_ತುಂಡು[r * ಅಡ್ಡಸಾಲು_stride + kt] : ಶೂನ್ಯಸೂಚಕ;
                    tile_dst[cp * 64 + 2 * ಅಡ್ಡಸಾಲು + 0] = b ? b->qs[col0] : 0;
                    tile_dst[cp * 64 + 2 * ಅಡ್ಡಸಾಲು + 1] = b ? b->qs[col1] : 0;
                }
            }
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < 32; ಅಡ್ಡಸಾಲು++) {
                ಸ್ಥಿರಾಂಕ int64_t r = ct * 32 + ಅಡ್ಡಸಾಲು;
                scale_dst[ಅಡ್ಡಸಾಲು] = (r < ne1 && kt < ne0 / 32) ? src_ತುಂಡು[r * ಅಡ್ಡಸಾಲು_stride + kt].d : 0;
            }
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ repack_tiled_q8_0(ಶೂನ್ಯ * ದತ್ತಾಂಶ, ಸ್ಥಿರಾಂಕ ggml_tensor * t, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    unpack_from_tiled_grid_32<ಖಂಡ_q8_0, 1>(ದತ್ತಾಂಶ, t, ಸ್ಥಾನಾಂತರ, size, HTP_TENSOR_TILE_ಗಾತ್ರ_Q8_0,
        [](ಖಂಡ_q8_0 * dst_ತುಂಡು, ಸ್ಥಿರಾಂಕ uint8_t * tile_src, int64_t ne0, int64_t ಆರಂಭ_ಅಡ್ಡಸಾಲು, int64_t ಕೊನೆ_ಅಡ್ಡಸಾಲು, int64_t ಅಡ್ಡಸಾಲು_stride, ಪೂರ್ಣಾಂಕ ct, ಪೂರ್ಣಾಂಕ kt) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ cp = 0; cp < 16; cp++) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ col0 = cp * 2;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ col1 = col0 + 1;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < 32; ಅಡ್ಡಸಾಲು++) {
                    ಸ್ಥಿರಾಂಕ int64_t r = ct * 32 + ಅಡ್ಡಸಾಲು;
                    ಆದರೆ (r >= ಆರಂಭ_ಅಡ್ಡಸಾಲು && r < ಕೊನೆ_ಅಡ್ಡಸಾಲು && kt < ne0 / 32) {
                        ಖಂಡ_q8_0 & b = dst_ತುಂಡು[(r - ಆರಂಭ_ಅಡ್ಡಸಾಲು) * ಅಡ್ಡಸಾಲು_stride + kt];
                        b.qs[col0] = tile_src[cp * 64 + 2 * ಅಡ್ಡಸಾಲು + 0];
                        b.qs[col1] = tile_src[cp * 64 + 2 * ಅಡ್ಡಸಾಲು + 1];
                    }
                }
            }
            ಸ್ಥಿರಾಂಕ ggml_half * scale_src = (ಸ್ಥಿರಾಂಕ ggml_half *) (tile_src + 1024);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < 32; ಅಡ್ಡಸಾಲು++) {
                ಸ್ಥಿರಾಂಕ int64_t r = ct * 32 + ಅಡ್ಡಸಾಲು;
                ಆದರೆ (r >= ಆರಂಭ_ಅಡ್ಡಸಾಲು && r < ಕೊನೆ_ಅಡ್ಡಸಾಲು && kt < ne0 / 32) {
                    dst_ತುಂಡು[(r - ಆರಂಭ_ಅಡ್ಡಸಾಲು) * ಅಡ್ಡಸಾಲು_stride + kt].d = scale_src[ಅಡ್ಡಸಾಲು];
                }
            }
        });
}

// --- Float Types (F16, F32, BF16 -> f16_tiled HMX layout) ---

ಸ್ಥಿರ ಶೂನ್ಯ repack_q2_0_tiled(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    GGML_ASSERT(ಸ್ಥಾನಾಂತರ == 0);
    GGML_ASSERT(t->ne[0] % 64 == 0);
    GGML_ASSERT(size == ggml_nbytes(t));

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_0 * src = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_0 *) ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ int64_t ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = t->ne[0] / 64;
    ಸ್ಥಿರಾಂಕ int64_t n_ಲಂಬಸಾಲು_tiles = hex_ಸುತ್ತು_ಮೇಲೆ(t->ne[1], 32) / 32;
    ಸ್ಥಿರಾಂಕ int64_t n_ತೂಕ_tiles = ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
    ಸ್ಥಿರಾಂಕ size_t ಮಾತೃಕೆ_ಗಾತ್ರ = (size_t) n_ಲಂಬಸಾಲು_tiles * n_ತೂಕ_tiles * HTP_TENSOR_TILE_ಗಾತ್ರ_Q2_0;

    ಗಾಗಿ (int64_t ತುಂಡು_2 = 0; ತುಂಡು_2 < t->ne[2] * t->ne[3]; ++ತುಂಡು_2) {
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_0 * src_ತುಂಡು = src + ತುಂಡು_2 * t->ne[1] * ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
        uint8_t * dst_ತುಂಡು = (uint8_t *) t->ದತ್ತಾಂಶ + ತುಂಡು_2 * ಮಾತೃಕೆ_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ size_t ct_ಬೈಟುಗಳು = (size_t) n_ತೂಕ_tiles * HTP_TENSOR_TILE_ಗಾತ್ರ_Q2_0;

        repack_parallel_for_ct(ಮಾತೃಕೆ_ಗಾತ್ರ, (ಪೂರ್ಣಾಂಕ) n_ಲಂಬಸಾಲು_tiles, [&](ಪೂರ್ಣಾಂಕ ct0, ಪೂರ್ಣಾಂಕ ct1) {
            memset(dst_ತುಂಡು + (size_t) ct0 * ct_ಬೈಟುಗಳು, 0, (size_t) (ct1 - ct0) * ct_ಬೈಟುಗಳು);
            ಸ್ಥಿರಾಂಕ int64_t r0 = (int64_t) ct0 * 32;
            ಸ್ಥಿರಾಂಕ int64_t r1 = (std::min)((int64_t) ct1 * 32, t->ne[1]);
            ಗಾಗಿ (int64_t ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ = r0; ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ < r1; ++ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ) {
                ಸ್ಥಿರಾಂಕ int64_t ct = ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ / 32;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = (ಪೂರ್ಣಾಂಕ) (ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ % 32);
                ಗಾಗಿ (int64_t kt = 0; kt < n_ತೂಕ_tiles; ++kt) {
                    ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_0 * ಖಂಡ = src_ತುಂಡು + ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ * ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + kt;
                    uint8_t * tile = dst_ತುಂಡು + (ct * n_ತೂಕ_tiles + kt) * HTP_TENSOR_TILE_ಗಾತ್ರ_Q2_0;
#ಆದರೆ defined(__ARM_NEON) || defined(__aarch64__)
                    repack_neon_q2_0_ಖಂಡ(ಖಂಡ->qs, tile, ಅಡ್ಡಸಾಲು);
#ಇಲ್ಲದಿದ್ದರೆ
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಗುಂಪು = 0; ಗುಂಪು < 16; ++ಗುಂಪು) {
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ byte = 0; byte < 4; ++byte) {
                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k = 4 * ಗುಂಪು + byte;
                            ಸ್ಥಿರಾಂಕ uint8_t code = (ಖಂಡ->qs[k / 4] >> (2 * (k % 4))) & 3;
                            tile[(ಗುಂಪು >> 2) * 128 + 4 * ಅಡ್ಡಸಾಲು + byte] |= code << (2 * (ಗುಂಪು & 3));
                        }
                    }
#ಆದರೆ_ಅಂತ್ಯ
                    ಸ್ಮರಣೆ_ನಕಲು(tile + 512 + 2 * ಅಡ್ಡಸಾಲು, &ಖಂಡ->d, ಗಾತ್ರ(ಖಂಡ->d));
                }
            }
        });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ repack_tiled_q2_0(ಶೂನ್ಯ * ದತ್ತಾಂಶ, ಸ್ಥಿರಾಂಕ ggml_tensor * t, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    GGML_ASSERT(t->ne[0] % 64 == 0);
    GGML_ASSERT(ಸ್ಥಾನಾಂತರ + size <= ggml_nbytes(t));

    ಸ್ಥಿರಾಂಕ size_t ಮೊದಲ_ಖಂಡ = ಸ್ಥಾನಾಂತರ / ಗಾತ್ರ(ಖಂಡ_q2_0);
    ಸ್ಥಿರಾಂಕ size_t ಕೊನೆ_ಖಂಡ = (ಸ್ಥಾನಾಂತರ + size + ಗಾತ್ರ(ಖಂಡ_q2_0) - 1) / ಗಾತ್ರ(ಖಂಡ_q2_0);
    ಸ್ಥಿರಾಂಕ size_t ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = t->ne[0] / 64;
    ಸ್ಥಿರಾಂಕ size_t ಖಂಡಗಳು_ಪ್ರತಿ_ತುಂಡು = t->ne[1] * ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
    ಸ್ಥಿರಾಂಕ size_t n_ತೂಕ_tiles = ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
    ಸ್ಥಿರಾಂಕ size_t n_ಲಂಬಸಾಲು_tiles = hex_ಸುತ್ತು_ಮೇಲೆ(t->ne[1], 32) / 32;
    ಸ್ಥಿರಾಂಕ size_t ಮಾತೃಕೆ_ಗಾತ್ರ = n_ಲಂಬಸಾಲು_tiles * n_ತೂಕ_tiles * HTP_TENSOR_TILE_ಗಾತ್ರ_Q2_0;

    ಗಾಗಿ (size_t ಖಂಡ_ಸೂಚ್ಯಂಕ = ಮೊದಲ_ಖಂಡ; ಖಂಡ_ಸೂಚ್ಯಂಕ < ಕೊನೆ_ಖಂಡ; ++ಖಂಡ_ಸೂಚ್ಯಂಕ) {
        ಸ್ಥಿರಾಂಕ size_t ತುಂಡು_2 = ಖಂಡ_ಸೂಚ್ಯಂಕ / ಖಂಡಗಳು_ಪ್ರತಿ_ತುಂಡು;
        ಸ್ಥಿರಾಂಕ size_t ಒಳ_ತುಂಡು = ಖಂಡ_ಸೂಚ್ಯಂಕ % ಖಂಡಗಳು_ಪ್ರತಿ_ತುಂಡು;
        ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ = ಒಳ_ತುಂಡು / ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
        ಸ್ಥಿರಾಂಕ size_t kt = ಒಳ_ತುಂಡು % ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
        ಸ್ಥಿರಾಂಕ size_t ct = ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ / 32;
        ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು = ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ % 32;
        ಸ್ಥಿರಾಂಕ uint8_t * tile = (ಸ್ಥಿರಾಂಕ uint8_t *) t->ದತ್ತಾಂಶ + ತುಂಡು_2 * ಮಾತೃಕೆ_ಗಾತ್ರ +
            (ct * n_ತೂಕ_tiles + kt) * HTP_TENSOR_TILE_ಗಾತ್ರ_Q2_0;

        ಖಂಡ_q2_0 ಖಂಡ = {};
        ಸ್ಮರಣೆ_ನಕಲು(&ಖಂಡ.d, tile + 512 + 2 * ಅಡ್ಡಸಾಲು, ಗಾತ್ರ(ಖಂಡ.d));
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಗುಂಪು = 0; ಗುಂಪು < 16; ++ಗುಂಪು) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ byte = 0; byte < 4; ++byte) {
                ಸ್ಥಿರಾಂಕ uint8_t packed = tile[(ಗುಂಪು >> 2) * 128 + 4 * ಅಡ್ಡಸಾಲು + byte];
                ಸ್ಥಿರಾಂಕ uint8_t code = (packed >> (2 * (ಗುಂಪು & 3))) & 3;
                ಖಂಡ.qs[ಗುಂಪು] |= code << (2 * byte);
            }
        }

        ಸ್ಥಿರಾಂಕ size_t ಖಂಡ_begin = ಖಂಡ_ಸೂಚ್ಯಂಕ * ಗಾತ್ರ(ಖಂಡ_q2_0);
        ಸ್ಥಿರಾಂಕ size_t copy_begin = (std::max)(ಖಂಡ_begin, ಸ್ಥಾನಾಂತರ);
        ಸ್ಥಿರಾಂಕ size_t copy_ಕೊನೆ = (std::min)(ಖಂಡ_begin + ಗಾತ್ರ(ಖಂಡ_q2_0), ಸ್ಥಾನಾಂತರ + size);
        ಸ್ಮರಣೆ_ನಕಲು((uint8_t *) ದತ್ತಾಂಶ + copy_begin - ಸ್ಥಾನಾಂತರ, (ಸ್ಥಿರಾಂಕ uint8_t *) &ಖಂಡ + copy_begin - ಖಂಡ_begin, copy_ಕೊನೆ - copy_begin);
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು SrcT, ಬಗೆ_ಹೆಸರು ಗಮ್ಯ_Fp16_Fn>
ಸ್ಥಿರ ಶೂನ್ಯ repack_float_ಗಮ್ಯ_f16_tiled(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size, ಗಮ್ಯ_Fp16_Fn && ಗಮ್ಯ_fp16) {
    repack_ಗಮ್ಯ_tiled_grid_32<SrcT, 32>(t, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, HTP_TENSOR_TILE_ಗಾತ್ರ_F16,
        [&](uint8_t * tile_ಬೈಟುಗಳು, ಸ್ಥಿರಾಂಕ SrcT * src_ತುಂಡು, int64_t ne0, int64_t ne1, int64_t ಅಡ್ಡಸಾಲು_stride, ಪೂರ್ಣಾಂಕ ct, ಪೂರ್ಣಾಂಕ kt) {
            uint16_t * tile_dst = (uint16_t *) tile_ಬೈಟುಗಳು;
#ಆದರೆ defined(__ARM_NEON) || defined(__aarch64__)
            ಆದರೆ (ct * 32 + 32 <= ne1 && kt * 32 + 32 <= ne0) {
                repack_neon_f16_tile(tile_dst, src_ತುಂಡು, ಅಡ್ಡಸಾಲು_stride, ct, kt);
                ಹಿಂತಿರುಗಿ;
            }
#ಆದರೆ_ಅಂತ್ಯ
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ cp = 0; cp < 16; cp++) {
                ಸ್ಥಿರಾಂಕ int64_t c0 = kt * 32 + cp * 2;
                ಸ್ಥಿರಾಂಕ int64_t c1 = c0 + 1;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < 32; ಅಡ್ಡಸಾಲು++) {
                    ಸ್ಥಿರಾಂಕ int64_t r = ct * 32 + ಅಡ್ಡಸಾಲು;
                    ಸ್ಥಿರಾಂಕ SrcT * r_ptr = (r < ne1) ? &src_ತುಂಡು[r * ಅಡ್ಡಸಾಲು_stride] : ಶೂನ್ಯಸೂಚಕ;
                    tile_dst[cp * 64 + ಅಡ್ಡಸಾಲು * 2 + 0] = (r_ptr && c0 < ne0) ? ಗಮ್ಯ_fp16(r_ptr[c0]) : 0;
                    tile_dst[cp * 64 + ಅಡ್ಡಸಾಲು * 2 + 1] = (r_ptr && c1 < ne0) ? ಗಮ್ಯ_fp16(r_ptr[c1]) : 0;
                }
            }
        });
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು DstT, ಬಗೆ_ಹೆಸರು FromFp16Fn>
ಸ್ಥಿರ ಶೂನ್ಯ unpack_f16_tiled_ಗಮ್ಯ_float(ಶೂನ್ಯ * ದತ್ತಾಂಶ, ಸ್ಥಿರಾಂಕ ggml_tensor * t, size_t ಸ್ಥಾನಾಂತರ, size_t size, FromFp16Fn && from_fp16) {
    unpack_from_tiled_grid_32<DstT, 32>(ದತ್ತಾಂಶ, t, ಸ್ಥಾನಾಂತರ, size, HTP_TENSOR_TILE_ಗಾತ್ರ_F16,
        [&](DstT * dst_ತುಂಡು, ಸ್ಥಿರಾಂಕ uint8_t * tile_ಬೈಟುಗಳು, int64_t ne0, int64_t ಆರಂಭ_ಅಡ್ಡಸಾಲು, int64_t ಕೊನೆ_ಅಡ್ಡಸಾಲು, int64_t ಅಡ್ಡಸಾಲು_stride, ಪೂರ್ಣಾಂಕ ct, ಪೂರ್ಣಾಂಕ kt) {
            ಸ್ಥಿರಾಂಕ uint16_t * tile_src = (ಸ್ಥಿರಾಂಕ uint16_t *) tile_ಬೈಟುಗಳು;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ cp = 0; cp < 16; cp++) {
                ಸ್ಥಿರಾಂಕ int64_t c0 = kt * 32 + cp * 2;
                ಸ್ಥಿರಾಂಕ int64_t c1 = c0 + 1;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < 32; ಅಡ್ಡಸಾಲು++) {
                    ಸ್ಥಿರಾಂಕ int64_t r = ct * 32 + ಅಡ್ಡಸಾಲು;
                    ಆದರೆ (r >= ಆರಂಭ_ಅಡ್ಡಸಾಲು && r < ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
                        DstT * r_ptr = dst_ತುಂಡು + (r - ಆರಂಭ_ಅಡ್ಡಸಾಲು) * ಅಡ್ಡಸಾಲು_stride;
                        ಆದರೆ (c0 < ne0) r_ptr[c0] = from_fp16(tile_src[cp * 64 + ಅಡ್ಡಸಾಲು * 2 + 0]);
                        ಆದರೆ (c1 < ne0) r_ptr[c1] = from_fp16(tile_src[cp * 64 + ಅಡ್ಡಸಾಲು * 2 + 1]);
                    }
                }
            }
        });
}

// --- Generic K-Quant (super-block 256) grid walkers ---

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು ಖಂಡ_T, ಬಗೆ_ಹೆಸರು PackKtFn>
ಸ್ಥಿರ ಶೂನ್ಯ repack_kquant_ಗಮ್ಯ_tiled_grid(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size,
                                        size_t tile_ಗಾತ್ರ, PackKtFn && pack_kt) {
    GGML_ASSERT(ಸ್ಥಾನಾಂತರ == 0);
    GGML_ಬಳಸದ(size);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_T * src_ಮಾತೃಕೆ = (ಸ್ಥಿರಾಂಕ ಖಂಡ_T *) ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ int64_t ne0 = t->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne1 = t->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne2 = t->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne3 = t->ne[3];
    GGML_ASSERT(ne0 % QK_K == 0);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ಲಂಬಸಾಲು_tiles    = hex_ಸುತ್ತು_ಮೇಲೆ(ne1, 32) / 32;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_k_tiles      = hex_ಸುತ್ತು_ಮೇಲೆ(ne0, 32) / 32;
    ಸ್ಥಿರಾಂಕ size_t ಮಾತೃಕೆ_ಗಾತ್ರ = (size_t) n_ಲಂಬಸಾಲು_tiles * n_k_tiles * tile_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ int64_t sb_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = ne0 / QK_K;

    ಗಾಗಿ (int64_t ತುಂಡು_idx = 0; ತುಂಡು_idx < ne2 * ne3; ತುಂಡು_idx++) {
        ಸ್ಥಿರಾಂಕ ಖಂಡ_T * src_ತುಂಡು = src_ಮಾತೃಕೆ + ತುಂಡು_idx * (ne1 * sb_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು);
        uint8_t * ಮಾತೃಕೆ_dst     = (uint8_t *) t->ದತ್ತಾಂಶ + ತುಂಡು_idx * ಮಾತೃಕೆ_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ size_t ct_ಬೈಟುಗಳು    = (size_t) n_k_tiles * tile_ಗಾತ್ರ;

        repack_parallel_for_ct(ಮಾತೃಕೆ_ಗಾತ್ರ, n_ಲಂಬಸಾಲು_tiles, [&](ಪೂರ್ಣಾಂಕ ct0, ಪೂರ್ಣಾಂಕ ct1) {
            // Padding rows and OR-ed bit planes require zeroed tiles
            memset(ಮಾತೃಕೆ_dst + (size_t) ct0 * ct_ಬೈಟುಗಳು, 0, (size_t) (ct1 - ct0) * ct_ಬೈಟುಗಳು);

            ಸ್ಥಿರಾಂಕ int64_t r0 = (int64_t) ct0 * 32;
            ಸ್ಥಿರಾಂಕ int64_t r1 = (std::min)((int64_t) ct1 * 32, ne1);
            ಗಾಗಿ (int64_t r = r0; r < r1; r++) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ct  = (ಪೂರ್ಣಾಂಕ) (r / 32);
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = (ಪೂರ್ಣಾಂಕ) (r % 32);
                ಸ್ಥಿರಾಂಕ ಖಂಡ_T * src_ಅಡ್ಡಸಾಲು = src_ತುಂಡು + r * sb_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile = ಮಾತೃಕೆ_dst + ((size_t) ct * n_k_tiles + kt) * tile_ಗಾತ್ರ;
                    pack_kt(tile, &src_ಅಡ್ಡಸಾಲು[kt / 8], ಅಡ್ಡಸಾಲು, kt % 8);
                }
            }
        });
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು ಖಂಡ_T, ಬಗೆ_ಹೆಸರು UnpackSbFn>
ಸ್ಥಿರ ಶೂನ್ಯ unpack_kquant_from_tiled_grid(ಶೂನ್ಯ * ದತ್ತಾಂಶ, ಸ್ಥಿರಾಂಕ ggml_tensor * t, size_t ಸ್ಥಾನಾಂತರ, size_t size,
                                          size_t tile_ಗಾತ್ರ, UnpackSbFn && unpack_sb) {
    uint8_t * dst_ಬೈಟುಗಳು = (uint8_t *) ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ int64_t ne0 = t->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne1 = t->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne2 = t->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne3 = t->ne[3];
    GGML_ASSERT(ne0 % QK_K == 0);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ಲಂಬಸಾಲು_tiles    = hex_ಸುತ್ತು_ಮೇಲೆ(ne1, 32) / 32;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_k_tiles      = hex_ಸುತ್ತು_ಮೇಲೆ(ne0, 32) / 32;
    ಸ್ಥಿರಾಂಕ size_t ಮಾತೃಕೆ_ಗಾತ್ರ = (size_t) n_ಲಂಬಸಾಲು_tiles * n_k_tiles * tile_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ int64_t sb_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = ne0 / QK_K;

    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು = ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(t->type, ne0);
    ಸ್ಥಿರಾಂಕ size_t ತುಂಡು_ಗಾತ್ರ     = (size_t) ne1 * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು;
    ಸ್ಥಿರಾಂಕ int64_t ಆರಂಭ_ತುಂಡು   = ಸ್ಥಾನಾಂತರ / ತುಂಡು_ಗಾತ್ರ;
    int64_t ಕೊನೆ_ತುಂಡು           = (ಸ್ಥಾನಾಂತರ + size + ತುಂಡು_ಗಾತ್ರ - 1) / ತುಂಡು_ಗಾತ್ರ;
    ಆದರೆ (ಕೊನೆ_ತುಂಡು > ne2 * ne3) {
        ಕೊನೆ_ತುಂಡು = ne2 * ne3;
    }

    ಗಾಗಿ (int64_t ತುಂಡು_idx = ಆರಂಭ_ತುಂಡು; ತುಂಡು_idx < ಕೊನೆ_ತುಂಡು; ತುಂಡು_idx++) {
        ಸ್ಥಿರಾಂಕ size_t cur_ಆರಂಭ_byte     = (std::max)(ಸ್ಥಾನಾಂತರ, (size_t) ತುಂಡು_idx * ತುಂಡು_ಗಾತ್ರ);
        ಸ್ಥಿರಾಂಕ size_t cur_ಕೊನೆ_byte       = (std::min)(ಸ್ಥಾನಾಂತರ + size, (size_t) (ತುಂಡು_idx + 1) * ತುಂಡು_ಗಾತ್ರ);
        ಸ್ಥಿರಾಂಕ size_t ತುಂಡು_ಸ್ಥಾನಾಂತರ_ಆರಂಭ = cur_ಆರಂಭ_byte - (size_t) ತುಂಡು_idx * ತುಂಡು_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ size_t ತುಂಡು_ಸ್ಥಾನಾಂತರ_ಕೊನೆ   = cur_ಕೊನೆ_byte - (size_t) ತುಂಡು_idx * ತುಂಡು_ಗಾತ್ರ;

        ಸ್ಥಿರಾಂಕ int64_t ಆರಂಭ_ಅಡ್ಡಸಾಲು = ತುಂಡು_ಸ್ಥಾನಾಂತರ_ಆರಂಭ / ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು;
        ಸ್ಥಿರಾಂಕ int64_t ಕೊನೆ_ಅಡ್ಡಸಾಲು   = (std::min)((int64_t) ((ತುಂಡು_ಸ್ಥಾನಾಂತರ_ಕೊನೆ + ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು - 1) / ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು), ne1);

        ಖಂಡ_T * dst_ತುಂಡು         = (ಖಂಡ_T *) (dst_ಬೈಟುಗಳು + (cur_ಆರಂಭ_byte - ಸ್ಥಾನಾಂತರ));
        ಸ್ಥಿರಾಂಕ uint8_t * ಮಾತೃಕೆ_src = (ಸ್ಥಿರಾಂಕ uint8_t *) t->ದತ್ತಾಂಶ + ತುಂಡು_idx * ಮಾತೃಕೆ_ಗಾತ್ರ;

        ಗಾಗಿ (int64_t r = ಆರಂಭ_ಅಡ್ಡಸಾಲು; r < ಕೊನೆ_ಅಡ್ಡಸಾಲು; r++) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ct  = (ಪೂರ್ಣಾಂಕ) (r / 32);
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = (ಪೂರ್ಣಾಂಕ) (r % 32);
            ಖಂಡ_T * dst_ಅಡ್ಡಸಾಲು = dst_ತುಂಡು + (r - ಆರಂಭ_ಅಡ್ಡಸಾಲು) * sb_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;

            ಗಾಗಿ (int64_t sb = 0; sb < sb_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು; sb++) {
                ಖಂಡ_T * b = &dst_ಅಡ್ಡಸಾಲು[sb];
                memset(b, 0, ಗಾತ್ರ(ಖಂಡ_T));
                ಸ್ಥಿರಾಂಕ uint8_t * sb_tiles = ಮಾತೃಕೆ_src + ((size_t) ct * n_k_tiles + sb * 8) * tile_ಗಾತ್ರ;
                unpack_sb(b, sb_tiles, tile_ಗಾತ್ರ, ಅಡ್ಡಸಾಲು);
            }
        }
    }
}

// --- K-Quant Helpers & Per-Type Implementations (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) ---

#ಆದರೆ !defined(__ARM_NEON) && !defined(__aarch64__)
// unsigned 6-bit value (0..63) of element e of a Q6_K block, same bit layout as dequantize_row_q6_K
ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint8_t q6_K_get_quant(ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K * b, ಪೂರ್ಣಾಂಕ e) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ c = e / 128;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ w = e % 128;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ g = w / 32;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ l = w % 32;
    ಸ್ಥಿರಾಂಕ uint8_t * ql = b->ql + c * 64;
    ಸ್ಥಿರಾಂಕ uint8_t * qh = b->qh + c * 32;
    uint8_t lo, hi;
    ಆಯ್ಕೆ (g) {
        ಸ್ಥಿತಿ 0:  lo = ql[l]      & 0xF; hi = (qh[l] >> 0) & 3; ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 1:  lo = ql[l + 32] & 0xF; hi = (qh[l] >> 2) & 3; ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2:  lo = ql[l]      >> 4;  hi = (qh[l] >> 4) & 3; ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: lo = ql[l + 32] >> 4;  hi = (qh[l] >> 6) & 3; ನಿಲ್ಲಿಸು;
    }
    ಹಿಂತಿರುಗಿ (uint8_t) (lo | (hi << 4));
}
#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ ಶೂನ್ಯ repack_q6_K_tiled(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    repack_kquant_ಗಮ್ಯ_tiled_grid<ಖಂಡ_q6_K>(t, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, HTP_TENSOR_TILE_ಗಾತ್ರ_Q6_K,
        [](uint8_t * tile, ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K * b, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು, ಪೂರ್ಣಾಂಕ kt_local) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d     = GGML_FP16_ಗಮ್ಯ_FP32(b->d);
            uint8_t * lo_pl   = tile;
            uint8_t * hi_pl   = tile + 512;
            ggml_half * sc_pl = (ggml_half *) (tile + 768);

#ಆದರೆ defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q6_k(b, kt_local, lo_pl, hi_pl, ಅಡ್ಡಸಾಲು);
#ಇಲ್ಲದಿದ್ದರೆ
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ lk = 0; lk < 32; lk++) {
                ಸ್ಥಿರಾಂಕ uint8_t q6 = q6_K_get_quant(b, kt_local * 32 + lk);
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ g   = lk >> 2;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ pos = ಅಡ್ಡಸಾಲು * 4 + (lk & 3);
                lo_pl[(g >> 1) * 128 + pos] |= (uint8_t) ((q6 & 0xF) << ((g & 1) * 4));
                hi_pl[(g >> 2) * 128 + pos] |= (uint8_t) ((q6 >> 4) << ((g & 3) * 2));
            }
#ಆದರೆ_ಅಂತ್ಯ
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + ಅಡ್ಡಸಾಲು] = GGML_FP32_ಗಮ್ಯ_FP16(d * (ದಶಮಾಂಶ) b->scales[kt_local * 2 + sub]);
            }
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ repack_tiled_q6_K(ಶೂನ್ಯ * ದತ್ತಾಂಶ, ಸ್ಥಿರಾಂಕ ggml_tensor * t, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    unpack_kquant_from_tiled_grid<ಖಂಡ_q6_K>(ದತ್ತಾಂಶ, t, ಸ್ಥಾನಾಂತರ, size, HTP_TENSOR_TILE_ಗಾತ್ರ_Q6_K,
        [](ಖಂಡ_q6_K * b, ಸ್ಥಿರಾಂಕ uint8_t * sb_tiles, size_t tile_ಗಾತ್ರ, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು) {
            ದಶಮಾಂಶ sub_scales[16];
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ kt_local = 0; kt_local < 8; kt_local++) {
                ಸ್ಥಿರಾಂಕ uint8_t *   tile  = sb_tiles + kt_local * tile_ಗಾತ್ರ;
                ಸ್ಥಿರಾಂಕ uint8_t *   lo_pl = tile;
                ಸ್ಥಿರಾಂಕ uint8_t *   hi_pl = tile + 512;
                ಸ್ಥಿರಾಂಕ ggml_half * sc_pl = (ಸ್ಥಿರಾಂಕ ggml_half *) (tile + 768);

                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ c  = kt_local / 4;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ g  = kt_local % 4;
                uint8_t * ql = b->ql + c * 64;
                uint8_t * qh = b->qh + c * 32;

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ lk = 0; lk < 32; lk++) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ g_tile = lk >> 2;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ pos    = ಅಡ್ಡಸಾಲು * 4 + (lk & 3);
                    ಸ್ಥಿರಾಂಕ uint8_t lo = (lo_pl[(g_tile >> 1) * 128 + pos] >> ((g_tile & 1) * 4)) & 0xF;
                    ಸ್ಥಿರಾಂಕ uint8_t hi = (hi_pl[(g_tile >> 2) * 128 + pos] >> ((g_tile & 3) * 2)) & 3;

                    ಆಯ್ಕೆ (g) {
                        ಸ್ಥಿತಿ 0:  ql[lk]      |= lo;        qh[lk] |= (hi << 0); ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ 1:  ql[lk + 32] |= lo;        qh[lk] |= (hi << 2); ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ 2:  ql[lk]      |= (lo << 4); qh[lk] |= (hi << 4); ನಿಲ್ಲಿಸು;
                        ಪೂರ್ವನಿಯೋಜಿತ: ql[lk + 32] |= (lo << 4); qh[lk] |= (hi << 6); ನಿಲ್ಲಿಸು;
                    }
                }

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ sub = 0; sub < 2; sub++) {
                    sub_scales[kt_local * 2 + sub] = GGML_FP16_ಗಮ್ಯ_FP32(sc_pl[sub * 32 + ಅಡ್ಡಸಾಲು]);
                }
            }

            ದಶಮಾಂಶ max_abs_scale = 0.0f;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 0; s < 16; s++) {
                max_abs_scale = (std::max)(max_abs_scale, fabsf(sub_scales[s]));
            }

            ಆದರೆ (max_abs_scale == 0.0f) {
                b->d = GGML_FP32_ಗಮ್ಯ_FP16(0.0f);
            } ಇಲ್ಲದಿದ್ದರೆ {
                b->d = GGML_FP32_ಗಮ್ಯ_FP16(max_abs_scale / 127.0f);
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d_actual = GGML_FP16_ಗಮ್ಯ_FP32(b->d);
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ inv_d    = (d_actual != 0.0f) ? (1.0f / d_actual) : 0.0f;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 0; s < 16; s++) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sc = (ಪೂರ್ಣಾಂಕ) roundf(sub_scales[s] * inv_d);
                    b->scales[s] = (int8_t) (std::max)(-128, (std::min)(127, sc));
                }
            }
        });
}

#ಆದರೆ !defined(__ARM_NEON) && !defined(__aarch64__)
// low 2 bits (0..3) of element e of a Q2_K or Q3_K block, same bit layout as dequantize_row_q2_K / q3_K
ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint8_t q2_3_K_get_low2(ಸ್ಥಿರಾಂಕ uint8_t * qs, ಪೂರ್ಣಾಂಕ e) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ c = e / 128;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j = (e % 128) / 32;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ l = e % 32;
    ಹಿಂತಿರುಗಿ (qs[c * 32 + l] >> (2 * j)) & 3;
}

// hmask bit of element e of a Q3_K block, same bit layout as dequantize_row_q3_K
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ q3_K_get_hbit(ಸ್ಥಿರಾಂಕ ಖಂಡ_q3_K * b, ಪೂರ್ಣಾಂಕ e) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ c = e / 128;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j = (e % 128) / 32;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ l = e % 32;
    ಹಿಂತಿರುಗಿ (b->hmask[l] >> (c * 4 + j)) & 1;
}
#ಆದರೆ_ಅಂತ್ಯ

// signed 6-bit scale j (-32..31) of a Q3_K block, same packing as quantize_row_q3_K_ref
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ q3_K_get_scale(ಸ್ಥಿರಾಂಕ uint8_t * scales, ಪೂರ್ಣಾಂಕ j) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ lo = (j < 8) ? (scales[j] & 0xF) : (scales[j - 8] >> 4);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ hi = (scales[8 + j % 4] >> (2 * (j / 4))) & 3;
    ಹಿಂತಿರುಗಿ (lo | (hi << 4)) - 32;
}

// read-back: find fp16 d and l[j] in [lmin, lmax] with fp16(d * l[j]) == prod[j] for all j, false if none
ಸ್ಥಿರ ತಾರ್ಕಿಕ hexagon_recover_k_scales(ಸ್ಥಿರಾಂಕ ggml_half * prod, ಪೂರ್ಣಾಂಕ n, ಪೂರ್ಣಾಂಕ lmin, ಪೂರ್ಣಾಂಕ lmax, ggml_half * d_ಹೊರಹೋಗು, ಪೂರ್ಣಾಂಕ * l_ಹೊರಹೋಗು) {
    ಪೂರ್ಣಾಂಕ jmax = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 1; j < n; j++) {
        ಆದರೆ (fabsf(GGML_FP16_ಗಮ್ಯ_FP32(prod[j])) > fabsf(GGML_FP16_ಗಮ್ಯ_FP32(prod[jmax]))) {
            jmax = j;
        }
    }
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ pmax = GGML_FP16_ಗಮ್ಯ_FP32(prod[jmax]);
    ಆದರೆ (pmax == 0.0f) {
        *d_ಹೊರಹೋಗು = GGML_FP32_ಗಮ್ಯ_FP16(0.0f);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < n; j++) {
            l_ಹೊರಹೋಗು[j] = 0;
        }
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    // the quantizers put the largest scale at or near the range end, so try large |l| first
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ lext = (std::max)(-lmin, lmax);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ a = lext; a >= 1; a--) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ sign : { 1, -1 }) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ lj = sign * a;
            ಆದರೆ (lj < lmin || lj > lmax) {
                ಮುಂದುವರಿ;
            }
            ಸ್ಥಿರಾಂಕ ggml_half d0 = GGML_FP32_ಗಮ್ಯ_FP16(pmax / (ದಶಮಾಂಶ) lj);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ulp : { 0, -1, 1 }) {
                ggml_half d = d0;
                uint16_t ಬಿಟ್ಟುಗಳು;
                ಸ್ಮರಣೆ_ನಕಲು(&ಬಿಟ್ಟುಗಳು, &d, ಗಾತ್ರ(ಬಿಟ್ಟುಗಳು));
                ಬಿಟ್ಟುಗಳು = (uint16_t) (ಬಿಟ್ಟುಗಳು + ulp);
                ಸ್ಮರಣೆ_ನಕಲು(&d, &ಬಿಟ್ಟುಗಳು, ಗಾತ್ರ(ಬಿಟ್ಟುಗಳು));

                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ df = GGML_FP16_ಗಮ್ಯ_FP32(d);
                ಆದರೆ (!std::isfinite(df) || df == 0.0f) {
                    ಮುಂದುವರಿ;
                }
                ತಾರ್ಕಿಕ ok = ನಿಜ;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < n && ok; j++) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ l = (ಪೂರ್ಣಾಂಕ) roundf(GGML_FP16_ಗಮ್ಯ_FP32(prod[j]) / df);
                    ಸ್ಥಿರಾಂಕ ggml_half p = GGML_FP32_ಗಮ್ಯ_FP16(df * (ದಶಮಾಂಶ) l);
                    ok = l >= lmin && l <= lmax && memcmp(&p, &prod[j], ಗಾತ್ರ(p)) == 0;
                    l_ಹೊರಹೋಗು[j] = l;
                }
                ಆದರೆ (ok) {
                    *d_ಹೊರಹೋಗು = d;
                    ಹಿಂತಿರುಗಿ ನಿಜ;
                }
            }
        }
    }
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

ಸ್ಥಿರ ಶೂನ್ಯ repack_q3_K_tiled(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    repack_kquant_ಗಮ್ಯ_tiled_grid<ಖಂಡ_q3_K>(t, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, HTP_TENSOR_TILE_ಗಾತ್ರ_Q3_K,
        [](uint8_t * tile, ಸ್ಥಿರಾಂಕ ಖಂಡ_q3_K * b, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು, ಪೂರ್ಣಾಂಕ kt_local) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d     = GGML_FP16_ಗಮ್ಯ_FP32(b->d);
            uint8_t * lo_pl   = tile;
            uint8_t * neg_pl  = tile + 256;
            ggml_half * sc_pl = (ggml_half *) (tile + 384);

#ಆದರೆ defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q2_3_k_low2(b->qs, kt_local, lo_pl, ಅಡ್ಡಸಾಲು);
            repack_neon_q3_k_hmask(b->hmask, kt_local, neg_pl, ಅಡ್ಡಸಾಲು);
#ಇಲ್ಲದಿದ್ದರೆ
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ lk = 0; lk < 32; lk++) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ e   = kt_local * 32 + lk;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ g   = lk >> 2;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ pos = ಅಡ್ಡಸಾಲು * 4 + (lk & 3);
                lo_pl[(g >> 2) * 128 + pos] |= (uint8_t) (q2_3_K_get_low2(b->qs, e) << ((g & 3) * 2));
                ಆದರೆ (!q3_K_get_hbit(b, e)) {
                    neg_pl[pos] |= (uint8_t) (1 << g);
                }
            }
#ಆದರೆ_ಅಂತ್ಯ
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + ಅಡ್ಡಸಾಲು] = GGML_FP32_ಗಮ್ಯ_FP16(d * (ದಶಮಾಂಶ) q3_K_get_scale(b->scales, kt_local * 2 + sub));
            }
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ repack_tiled_q3_K(ಶೂನ್ಯ * ದತ್ತಾಂಶ, ಸ್ಥಿರಾಂಕ ggml_tensor * t, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    unpack_kquant_from_tiled_grid<ಖಂಡ_q3_K>(ದತ್ತಾಂಶ, t, ಸ್ಥಾನಾಂತರ, size, HTP_TENSOR_TILE_ಗಾತ್ರ_Q3_K,
        [](ಖಂಡ_q3_K * b, ಸ್ಥಿರಾಂಕ uint8_t * sb_tiles, size_t tile_ಗಾತ್ರ, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು) {
            ggml_half sub_scales[16];
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ kt_local = 0; kt_local < 8; kt_local++) {
                ಸ್ಥಿರಾಂಕ uint8_t *   tile   = sb_tiles + kt_local * tile_ಗಾತ್ರ;
                ಸ್ಥಿರಾಂಕ uint8_t *   lo_pl  = tile;
                ಸ್ಥಿರಾಂಕ uint8_t *   neg_pl = tile + 256;
                ಸ್ಥಿರಾಂಕ ggml_half * sc_pl  = (ಸ್ಥಿರಾಂಕ ggml_half *) (tile + 384);

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ lk = 0; lk < 32; lk++) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ e   = kt_local * 32 + lk;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ g   = lk >> 2;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ pos = ಅಡ್ಡಸಾಲು * 4 + (lk & 3);
                    ಸ್ಥಿರಾಂಕ uint8_t lo = (lo_pl[(g >> 2) * 128 + pos] >> ((g & 3) * 2)) & 3;

                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ c = e / 128;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j = (e % 128) / 32;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ l = e % 32;
                    b->qs[c * 32 + l] |= (uint8_t) (lo << (2 * j));
                    ಆದರೆ (!((neg_pl[pos] >> g) & 1)) {
                        b->hmask[l] |= (uint8_t) (1 << (c * 4 + j));
                    }
                }

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ sub = 0; sub < 2; sub++) {
                    sub_scales[kt_local * 2 + sub] = sc_pl[sub * 32 + ಅಡ್ಡಸಾಲು];
                }
            }

            ಪೂರ್ಣಾಂಕ ls[16];
            ಆದರೆ (!hexagon_recover_k_scales(sub_scales, 16, -32, 31, &b->d, ls)) {
                ದಶಮಾಂಶ max_scale = 0.0f;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 0; s < 16; s++) {
                    ಆದರೆ (fabsf(GGML_FP16_ಗಮ್ಯ_FP32(sub_scales[s])) > fabsf(max_scale)) {
                        max_scale = GGML_FP16_ಗಮ್ಯ_FP32(sub_scales[s]);
                    }
                }
                b->d = GGML_FP32_ಗಮ್ಯ_FP16(-max_scale / 32.0f);
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d_actual = GGML_FP16_ಗಮ್ಯ_FP32(b->d);
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ inv_d    = (d_actual != 0.0f) ? (1.0f / d_actual) : 0.0f;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 0; s < 16; s++) {
                    ls[s] = (std::max)(-32, (std::min)(31, (ಪೂರ್ಣಾಂಕ) roundf(GGML_FP16_ಗಮ್ಯ_FP32(sub_scales[s]) * inv_d)));
                }
            }

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 0; s < 16; s++) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ l = ls[s] + 32;
                ಆದರೆ (s < 8) {
                    b->scales[s] = l & 0xF;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    b->scales[s - 8] |= (uint8_t) ((l & 0xF) << 4);
                }
                b->scales[s % 4 + 8] |= (uint8_t) ((l >> 4) << (2 * (s / 4)));
            }
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ repack_q2_K_tiled(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    repack_kquant_ಗಮ್ಯ_tiled_grid<ಖಂಡ_q2_K>(t, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, HTP_TENSOR_TILE_ಗಾತ್ರ_Q2_K,
        [](uint8_t * tile, ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_K * b, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು, ಪೂರ್ಣಾಂಕ kt_local) {
            uint8_t * lo_pl   = tile;
            uint8_t * sc_pl   = tile + 256;
            ggml_half * dd_pl = (ggml_half *) (tile + 320);

#ಆದರೆ defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q2_3_k_low2(b->qs, kt_local, lo_pl, ಅಡ್ಡಸಾಲು);
#ಇಲ್ಲದಿದ್ದರೆ
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ lk = 0; lk < 32; lk++) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ g   = lk >> 2;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ pos = ಅಡ್ಡಸಾಲು * 4 + (lk & 3);
                lo_pl[(g >> 2) * 128 + pos] |= (uint8_t) (q2_3_K_get_low2(b->qs, kt_local * 32 + lk) << ((g & 3) * 2));
            }
#ಆದರೆ_ಅಂತ್ಯ
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + ಅಡ್ಡಸಾಲು] = b->scales[kt_local * 2 + sub];
            }
            ಆದರೆ (ಅಡ್ಡಸಾಲು / 8 == kt_local % 4) {
                dd_pl[ಅಡ್ಡಸಾಲು % 8] = kt_local < 4 ? b->d : (ggml_half) (b->dmin ^ 0x8000);
            }
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ repack_tiled_q2_K(ಶೂನ್ಯ * ದತ್ತಾಂಶ, ಸ್ಥಿರಾಂಕ ggml_tensor * t, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    unpack_kquant_from_tiled_grid<ಖಂಡ_q2_K>(ದತ್ತಾಂಶ, t, ಸ್ಥಾನಾಂತರ, size, HTP_TENSOR_TILE_ಗಾತ್ರ_Q2_K,
        [](ಖಂಡ_q2_K * b, ಸ್ಥಿರಾಂಕ uint8_t * sb_tiles, size_t tile_ಗಾತ್ರ, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ kt_local = 0; kt_local < 8; kt_local++) {
                ಸ್ಥಿರಾಂಕ uint8_t *   tile  = sb_tiles + kt_local * tile_ಗಾತ್ರ;
                ಸ್ಥಿರಾಂಕ uint8_t *   lo_pl = tile;
                ಸ್ಥಿರಾಂಕ uint8_t *   sc_pl = tile + 256;
                ಸ್ಥಿರಾಂಕ ggml_half * dd_pl = (ಸ್ಥಿರಾಂಕ ggml_half *) (tile + 320);

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ lk = 0; lk < 32; lk++) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ e   = kt_local * 32 + lk;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ g   = lk >> 2;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ pos = ಅಡ್ಡಸಾಲು * 4 + (lk & 3);
                    ಸ್ಥಿರಾಂಕ uint8_t lo = (lo_pl[(g >> 2) * 128 + pos] >> ((g & 3) * 2)) & 3;
                    b->qs[(e / 128) * 32 + e % 32] |= (uint8_t) (lo << (2 * ((e % 128) / 32)));
                }

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ sub = 0; sub < 2; sub++) {
                    b->scales[kt_local * 2 + sub] = sc_pl[sub * 32 + ಅಡ್ಡಸಾಲು];
                }
                ಆದರೆ (ಅಡ್ಡಸಾಲು / 8 == kt_local % 4) {
                    ಆದರೆ (kt_local < 4) {
                        b->d = dd_pl[ಅಡ್ಡಸಾಲು % 8];
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        b->dmin = (ggml_half) (dd_pl[ಅಡ್ಡಸಾಲು % 8] ^ 0x8000);
                    }
                }
            }
        });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ get_scale_min_k4(ಪೂರ್ಣಾಂಕ j, ಸ್ಥಿರಾಂಕ uint8_t * q, uint8_t * d, uint8_t * m) {
    ಆದರೆ (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } ಇಲ್ಲದಿದ್ದರೆ {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ pack_k4_scales_mins(uint8_t * scales, ggml_half * d, ggml_half * dmin,
                                       ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * sub_scales, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * sub_mins) {
    ದಶಮಾಂಶ max_scale = 0.0f;
    ದಶಮಾಂಶ max_min   = 0.0f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; j++) {
        ಆದರೆ (sub_scales[j] > max_scale) max_scale = sub_scales[j];
        ಆದರೆ (sub_mins[j]   > max_min)   max_min   = sub_mins[j];
    }

    ದಶಮಾಂಶ inv_scale = 0.0f;
    ಆದರೆ (max_scale > 0.0f) {
        *d = GGML_FP32_ಗಮ್ಯ_FP16(max_scale / 63.0f);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d_actual = GGML_FP16_ಗಮ್ಯ_FP32(*d);
        inv_scale = (d_actual > 0.0f) ? (1.0f / d_actual) : 0.0f;
    } ಇಲ್ಲದಿದ್ದರೆ {
        *d = GGML_FP32_ಗಮ್ಯ_FP16(0.0f);
    }

    ದಶಮಾಂಶ inv_min = 0.0f;
    ಆದರೆ (max_min > 0.0f) {
        *dmin = GGML_FP32_ಗಮ್ಯ_FP16(max_min / 63.0f);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin_actual = GGML_FP16_ಗಮ್ಯ_FP32(*dmin);
        inv_min = (dmin_actual > 0.0f) ? (1.0f / dmin_actual) : 0.0f;
    } ಇಲ್ಲದಿದ್ದರೆ {
        *dmin = GGML_FP32_ಗಮ್ಯ_FP16(0.0f);
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; j++) {
        uint8_t ls = (std::min)((uint8_t) 63, (uint8_t) roundf(inv_scale * sub_scales[j]));
        uint8_t lm = (std::min)((uint8_t) 63, (uint8_t) roundf(inv_min * sub_mins[j]));
        ಆದರೆ (j < 4) {
            scales[j]     = ls;
            scales[j + 4] = lm;
        } ಇಲ್ಲದಿದ್ದರೆ {
            scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            scales[j - 4] |= ((ls >> 4) << 6);
            scales[j - 0] |= ((lm >> 4) << 6);
        }
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು ಖಂಡ_T>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ repack_k4_quants(uint8_t * tile_dst, ಸ್ಥಿರಾಂಕ ಖಂಡ_T * b, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು, ಪೂರ್ಣಾಂಕ kt_local,
                                    int64_t sb_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, ತಾರ್ಕಿಕ use_neon) {
#ಆದರೆ defined(__ARM_NEON) || defined(__aarch64__)
    ಆದರೆ (use_neon) {
        ಆದರೆ ((ಅಡ್ಡಸಾಲು & 7) == 0) {
            ಸ್ಥಿರಾಂಕ uint8_t * qs[8];
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ rr = 0; rr < 8; ++rr) {
                qs[rr] = b[rr * sb_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು].qs + (kt_local / 2) * 32;
            }
            repack_neon_k4_ಅಡ್ಡಸಾಲುಗಳು(qs, tile_dst, ಅಡ್ಡಸಾಲು, (kt_local & 1) != 0);
        }
        ಹಿಂತಿರುಗಿ;
    }
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ಬಳಸದ(sb_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು);
    GGML_ಬಳಸದ(use_neon);
#ಆದರೆ_ಅಂತ್ಯ
    ಸ್ಥಿರಾಂಕ uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸರಿಸು = (kt_local & 1) ? 4 : 0;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ cp = 0; cp < 16; cp++) {
        ಸ್ಥಿರಾಂಕ uint8_t q0 = (qs_sub[2 * cp + 0] >> ಸರಿಸು) & 0x0F;
        ಸ್ಥಿರಾಂಕ uint8_t q1 = (qs_sub[2 * cp + 1] >> ಸರಿಸು) & 0x0F;
        tile_dst[cp * 32 + ಅಡ್ಡಸಾಲು] = (uint8_t) ((q1 << 4) | q0);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ repack_q4_K_tiled(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    ಸ್ಥಿರಾಂಕ int64_t sb_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = t->ne[0] / QK_K;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ    use_neon   = (t->ne[1] % 8 == 0);
    repack_kquant_ಗಮ್ಯ_tiled_grid<ಖಂಡ_q4_K>(t, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, HTP_TENSOR_TILE_ಗಾತ್ರ_Q4_K,
        [sb_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, use_neon](uint8_t * tile_dst, ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * b, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು, ಪೂರ್ಣಾಂಕ kt_local) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d    = GGML_FP16_ಗಮ್ಯ_FP32(b->d);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin = GGML_FP16_ಗಮ್ಯ_FP32(b->dmin);

            uint8_t sc, m;
            get_scale_min_k4(kt_local, b->scales, &sc, &m);

            repack_k4_quants(tile_dst, b, ಅಡ್ಡಸಾಲು, kt_local, sb_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, use_neon);

            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
            scale_dst[2 * ಅಡ್ಡಸಾಲು + 0] = GGML_FP32_ಗಮ್ಯ_FP16(d * (ದಶಮಾಂಶ) sc);
            scale_dst[2 * ಅಡ್ಡಸಾಲು + 1] = GGML_FP32_ಗಮ್ಯ_FP16(-dmin * (ದಶಮಾಂಶ) m);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ repack_tiled_q4_K(ಶೂನ್ಯ * ದತ್ತಾಂಶ, ಸ್ಥಿರಾಂಕ ggml_tensor * t, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    unpack_kquant_from_tiled_grid<ಖಂಡ_q4_K>(ದತ್ತಾಂಶ, t, ಸ್ಥಾನಾಂತರ, size, HTP_TENSOR_TILE_ಗಾತ್ರ_Q4_K,
        [](ಖಂಡ_q4_K * b, ಸ್ಥಿರಾಂಕ uint8_t * sb_tiles, size_t tile_ಗಾತ್ರ, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು) {
            ದಶಮಾಂಶ sub_scales[8];
            ದಶಮಾಂಶ sub_mins[8];

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ kt_local = 0; kt_local < 8; kt_local++) {
                ಸ್ಥಿರಾಂಕ uint8_t *   tile_src  = sb_tiles + kt_local * tile_ಗಾತ್ರ;
                ಸ್ಥಿರಾಂಕ ggml_half * scale_src = (ಸ್ಥಿರಾಂಕ ggml_half *) (tile_src + 512);

                uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸರಿಸು  = (kt_local & 1) ? 4 : 0;

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ cp = 0; cp < 16; cp++) {
                    ಸ್ಥಿರಾಂಕ uint8_t val = tile_src[cp * 32 + ಅಡ್ಡಸಾಲು];
                    qs_sub[2 * cp + 0] |= (uint8_t) ((val & 0x0F) << ಸರಿಸು);
                    qs_sub[2 * cp + 1] |= (uint8_t) ((val >> 4)   << ಸರಿಸು);
                }

                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ D = GGML_FP16_ಗಮ್ಯ_FP32(scale_src[2 * ಅಡ್ಡಸಾಲು + 0]);
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ M = GGML_FP16_ಗಮ್ಯ_FP32(scale_src[2 * ಅಡ್ಡಸಾಲು + 1]);
                sub_scales[kt_local] = (D > 0.0f) ? D : 0.0f;
                sub_mins[kt_local]   = (-M > 0.0f) ? -M : 0.0f;
            }

            pack_k4_scales_mins(b->scales, &b->d, &b->dmin, sub_scales, sub_mins);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ repack_q5_K_tiled(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    ಸ್ಥಿರಾಂಕ int64_t sb_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = t->ne[0] / QK_K;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ    use_neon   = (t->ne[1] % 8 == 0);
    repack_kquant_ಗಮ್ಯ_tiled_grid<ಖಂಡ_q5_K>(t, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, HTP_TENSOR_TILE_ಗಾತ್ರ_Q5_K,
        [sb_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, use_neon](uint8_t * tile_dst, ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K * b, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು, ಪೂರ್ಣಾಂಕ kt_local) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d     = GGML_FP16_ಗಮ್ಯ_FP32(b->d);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin  = GGML_FP16_ಗಮ್ಯ_FP32(b->dmin);
            uint8_t * plane   = tile_dst + 640;

            uint8_t sc, m;
            get_scale_min_k4(kt_local, b->scales, &sc, &m);

            repack_k4_quants(tile_dst, b, ಅಡ್ಡಸಾಲು, kt_local, sb_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, use_neon);

            ಸ್ಥಿರಾಂಕ uint8_t hbit = (uint8_t) (1 << kt_local);
#ಆದರೆ defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q5_k_qh(b->qh, hbit, plane, ಅಡ್ಡಸಾಲು);
#ಇಲ್ಲದಿದ್ದರೆ
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ cp = 0; cp < 16; cp++) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i    = cp / 4;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ lane = (cp % 4) * 32 + ಅಡ್ಡಸಾಲು;
                ಆದರೆ (b->qh[2 * cp + 0] & hbit) {
                    plane[lane] |= (uint8_t) (1 << (2 * i));
                }
                ಆದರೆ (b->qh[2 * cp + 1] & hbit) {
                    plane[lane] |= (uint8_t) (1 << (2 * i + 1));
                }
            }
#ಆದರೆ_ಅಂತ್ಯ

            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
            scale_dst[2 * ಅಡ್ಡಸಾಲು + 0] = GGML_FP32_ಗಮ್ಯ_FP16(d * (ದಶಮಾಂಶ) sc);
            scale_dst[2 * ಅಡ್ಡಸಾಲು + 1] = GGML_FP32_ಗಮ್ಯ_FP16(-dmin * (ದಶಮಾಂಶ) m);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ repack_tiled_q5_K(ಶೂನ್ಯ * ದತ್ತಾಂಶ, ಸ್ಥಿರಾಂಕ ggml_tensor * t, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    unpack_kquant_from_tiled_grid<ಖಂಡ_q5_K>(ದತ್ತಾಂಶ, t, ಸ್ಥಾನಾಂತರ, size, HTP_TENSOR_TILE_ಗಾತ್ರ_Q5_K,
        [](ಖಂಡ_q5_K * b, ಸ್ಥಿರಾಂಕ uint8_t * sb_tiles, size_t tile_ಗಾತ್ರ, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು) {
            ದಶಮಾಂಶ sub_scales[8];
            ದಶಮಾಂಶ sub_mins[8];

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ kt_local = 0; kt_local < 8; kt_local++) {
                ಸ್ಥಿರಾಂಕ uint8_t *   tile_src  = sb_tiles + kt_local * tile_ಗಾತ್ರ;
                ಸ್ಥಿರಾಂಕ uint8_t *   plane     = tile_src + 640;
                ಸ್ಥಿರಾಂಕ ggml_half * scale_src = (ಸ್ಥಿರಾಂಕ ggml_half *) (tile_src + 512);

                uint8_t * qs_sub   = b->qs + (kt_local / 2) * 32;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸರಿಸು    = (kt_local & 1) ? 4 : 0;
                ಸ್ಥಿರಾಂಕ uint8_t hbit = (uint8_t) (1 << kt_local);

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ cp = 0; cp < 16; cp++) {
                    ಸ್ಥಿರಾಂಕ uint8_t val = tile_src[cp * 32 + ಅಡ್ಡಸಾಲು];
                    qs_sub[2 * cp + 0] |= (uint8_t) ((val & 0x0F) << ಸರಿಸು);
                    qs_sub[2 * cp + 1] |= (uint8_t) ((val >> 4)   << ಸರಿಸು);

                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i    = cp / 4;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ lane = (cp % 4) * 32 + ಅಡ್ಡಸಾಲು;
                    ಆದರೆ (plane[lane] & (1 << (2 * i))) {
                        b->qh[2 * cp + 0] |= hbit;
                    }
                    ಆದರೆ (plane[lane] & (1 << (2 * i + 1))) {
                        b->qh[2 * cp + 1] |= hbit;
                    }
                }

                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ D = GGML_FP16_ಗಮ್ಯ_FP32(scale_src[2 * ಅಡ್ಡಸಾಲು + 0]);
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ M = GGML_FP16_ಗಮ್ಯ_FP32(scale_src[2 * ಅಡ್ಡಸಾಲು + 1]);
                sub_scales[kt_local] = (D > 0.0f) ? D : 0.0f;
                sub_mins[kt_local]   = (-M > 0.0f) ? -M : 0.0f;
            }

            pack_k4_scales_mins(b->scales, &b->d, &b->dmin, sub_scales, sub_mins);
        });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ q1_0_tiled_ಬಿಟ್_location(ಪೂರ್ಣಾಂಕ k, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು, ಪೂರ್ಣಾಂಕ & byte_ಸ್ಥಾನಾಂತರ, ಪೂರ್ಣಾಂಕ & ಬಿಟ್) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ quarter = k / 32;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kq      = k % 32;

    byte_ಸ್ಥಾನಾಂತರ = quarter * 128 + 4 * ಅಡ್ಡಸಾಲು + (kq % 4);
    ಬಿಟ್         = kq / 4;
}

ಸ್ಥಿರ ಶೂನ್ಯ repack_q1_0_tiled(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q1_0 * src_ಮಾತೃಕೆ = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q1_0 *) ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ int64_t ne0 = t->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne1 = t->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne2 = t->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne3 = t->ne[3];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ಲಂಬಸಾಲು_tiles = hex_ಸುತ್ತು_ಮೇಲೆ(ne1, 32) / 32;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_k_tiles = ne0 / QK1_0;
    ಸ್ಥಿರಾಂಕ size_t tile_ಗಾತ್ರ = HTP_TENSOR_TILE_ಗಾತ್ರ_Q1_0;
    ಸ್ಥಿರಾಂಕ size_t ಮಾತೃಕೆ_ಗಾತ್ರ = (size_t) n_ಲಂಬಸಾಲು_tiles * n_k_tiles * tile_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t ತುಂಡು_ಗಾತ್ರ = ne1 * ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(t->type, ne0);
    ಸ್ಥಿರಾಂಕ int64_t ಆರಂಭ_ತುಂಡು = ಸ್ಥಾನಾಂತರ / ತುಂಡು_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ int64_t ಕೊನೆ_ತುಂಡು = (std::min)((int64_t) ((ಸ್ಥಾನಾಂತರ + size + ತುಂಡು_ಗಾತ್ರ - 1) / ತುಂಡು_ಗಾತ್ರ), ne2 * ne3);

    GGML_ASSERT(ne0 % QK1_0 == 0);

    ಗಾಗಿ (int64_t ತುಂಡು_idx = ಆರಂಭ_ತುಂಡು; ತುಂಡು_idx < ಕೊನೆ_ತುಂಡು; ತುಂಡು_idx++) {
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q1_0 * src_ತುಂಡು = src_ಮಾತೃಕೆ + (ತುಂಡು_idx - ಆರಂಭ_ತುಂಡು) * ne1 * n_k_tiles;
        uint8_t * ಮಾತೃಕೆ_dst = (uint8_t *) t->ದತ್ತಾಂಶ + ತುಂಡು_idx * ಮಾತೃಕೆ_ಗಾತ್ರ;

        repack_parallel_for_ct(ಮಾತೃಕೆ_ಗಾತ್ರ, n_ಲಂಬಸಾಲು_tiles, [&](ಪೂರ್ಣಾಂಕ ct0, ಪೂರ್ಣಾಂಕ ct1) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ct = ct0; ct < ct1; ct++) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = ಮಾತೃಕೆ_dst + ((size_t) ct * n_k_tiles + kt) * tile_ಗಾತ್ರ;
                    ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
                    memset(tile_dst, 0, tile_ಗಾತ್ರ);

                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < 32; ಅಡ್ಡಸಾಲು++) {
                        ಸ್ಥಿರಾಂಕ int64_t r = ct * 32 + ಅಡ್ಡಸಾಲು;
                        ಆದರೆ (r >= ne1) {
                            ಮುಂದುವರಿ;
                        }

                        ಸ್ಥಿರಾಂಕ ಖಂಡ_q1_0 & ಖಂಡ = src_ತುಂಡು[r * n_k_tiles + kt];
                        scale_dst[ಅಡ್ಡಸಾಲು] = ಖಂಡ.d;
#ಆದರೆ defined(__ARM_NEON) || defined(__aarch64__)
                        repack_neon_q1_0_ಖಂಡ(ಖಂಡ.qs, tile_dst, ಅಡ್ಡಸಾಲು);
#ಇಲ್ಲದಿದ್ದರೆ
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < QK1_0; k++) {
                            ಪೂರ್ಣಾಂಕ byte_ಸ್ಥಾನಾಂತರ;
                            ಪೂರ್ಣಾಂಕ ಬಿಟ್;
                            q1_0_tiled_ಬಿಟ್_location(k, ಅಡ್ಡಸಾಲು, byte_ಸ್ಥಾನಾಂತರ, ಬಿಟ್);
                            ಆದರೆ (ಖಂಡ.qs[k / 8] & (1u << (k % 8))) {
                                tile_dst[byte_ಸ್ಥಾನಾಂತರ] |= (uint8_t) (1u << ಬಿಟ್);
                            }
                        }
#ಆದರೆ_ಅಂತ್ಯ
                    }
                }
            }
        });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ repack_tiled_q1_0(ಶೂನ್ಯ * ದತ್ತಾಂಶ, ಸ್ಥಿರಾಂಕ ggml_tensor * t, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    ಖಂಡ_q1_0 * dst_ಮಾತೃಕೆ = (ಖಂಡ_q1_0 *) ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ int64_t ne0 = t->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne1 = t->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne2 = t->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne3 = t->ne[3];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ಲಂಬಸಾಲು_tiles = hex_ಸುತ್ತು_ಮೇಲೆ(ne1, 32) / 32;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_k_tiles = ne0 / QK1_0;
    ಸ್ಥಿರಾಂಕ size_t tile_ಗಾತ್ರ = HTP_TENSOR_TILE_ಗಾತ್ರ_Q1_0;
    ಸ್ಥಿರಾಂಕ size_t ಮಾತೃಕೆ_ಗಾತ್ರ = (size_t) n_ಲಂಬಸಾಲು_tiles * n_k_tiles * tile_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು = ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(t->type, ne0);
    ಸ್ಥಿರಾಂಕ size_t ತುಂಡು_ಗಾತ್ರ = ne1 * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು;
    ಸ್ಥಿರಾಂಕ int64_t ಆರಂಭ_ತುಂಡು = ಸ್ಥಾನಾಂತರ / ತುಂಡು_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ int64_t ಕೊನೆ_ತುಂಡು = (std::min)((int64_t) ((ಸ್ಥಾನಾಂತರ + size + ತುಂಡು_ಗಾತ್ರ - 1) / ತುಂಡು_ಗಾತ್ರ), ne2 * ne3);

    GGML_ASSERT(ne0 % QK1_0 == 0);

    ಗಾಗಿ (int64_t ತುಂಡು_idx = ಆರಂಭ_ತುಂಡು; ತುಂಡು_idx < ಕೊನೆ_ತುಂಡು; ತುಂಡು_idx++) {
        ಸ್ಥಿರಾಂಕ size_t cur_ಆರಂಭ_byte = (std::max)(ಸ್ಥಾನಾಂತರ, (size_t) ತುಂಡು_idx * ತುಂಡು_ಗಾತ್ರ);
        ಸ್ಥಿರಾಂಕ size_t cur_ಕೊನೆ_byte = (std::min)(ಸ್ಥಾನಾಂತರ + size, (size_t) (ತುಂಡು_idx + 1) * ತುಂಡು_ಗಾತ್ರ);
        ಸ್ಥಿರಾಂಕ int64_t ಆರಂಭ_ಅಡ್ಡಸಾಲು = (cur_ಆರಂಭ_byte - (size_t) ತುಂಡು_idx * ತುಂಡು_ಗಾತ್ರ) / ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು;
        ಸ್ಥಿರಾಂಕ int64_t ಕೊನೆ_ಅಡ್ಡಸಾಲು = (std::min)((int64_t) ((cur_ಕೊನೆ_byte - (size_t) ತುಂಡು_idx * ತುಂಡು_ಗಾತ್ರ + ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು - 1) / ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು), ne1);
        ಖಂಡ_q1_0 * dst_ತುಂಡು = dst_ಮಾತೃಕೆ + (cur_ಆರಂಭ_byte - ಸ್ಥಾನಾಂತರ) / ಗಾತ್ರ(ಖಂಡ_q1_0);
        ಸ್ಥಿರಾಂಕ uint8_t * ಮಾತೃಕೆ_src = (ಸ್ಥಿರಾಂಕ uint8_t *) t->ದತ್ತಾಂಶ + ತುಂಡು_idx * ಮಾತೃಕೆ_ಗಾತ್ರ;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ct = ಆರಂಭ_ಅಡ್ಡಸಾಲು / 32; ct < (ಕೊನೆ_ಅಡ್ಡಸಾಲು + 31) / 32; ct++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ kt = 0; kt < n_k_tiles; kt++) {
                ಸ್ಥಿರಾಂಕ uint8_t * tile_src = ಮಾತೃಕೆ_src + ((size_t) ct * n_k_tiles + kt) * tile_ಗಾತ್ರ;
                ಸ್ಥಿರಾಂಕ ggml_half * scale_src = (ಸ್ಥಿರಾಂಕ ggml_half *) (tile_src + 512);

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < 32; ಅಡ್ಡಸಾಲು++) {
                    ಸ್ಥಿರಾಂಕ int64_t r = ct * 32 + ಅಡ್ಡಸಾಲು;
                    ಆದರೆ (r < ಆರಂಭ_ಅಡ್ಡಸಾಲು || r >= ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
                        ಮುಂದುವರಿ;
                    }

                    ಖಂಡ_q1_0 & ಖಂಡ = dst_ತುಂಡು[(r - ಆರಂಭ_ಅಡ್ಡಸಾಲು) * n_k_tiles + kt];
                    memset(ಖಂಡ.qs, 0, ಗಾತ್ರ(ಖಂಡ.qs));
                    ಖಂಡ.d = scale_src[ಅಡ್ಡಸಾಲು];
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < QK1_0; k++) {
                        ಪೂರ್ಣಾಂಕ byte_ಸ್ಥಾನಾಂತರ;
                        ಪೂರ್ಣಾಂಕ ಬಿಟ್;
                        q1_0_tiled_ಬಿಟ್_location(k, ಅಡ್ಡಸಾಲು, byte_ಸ್ಥಾನಾಂತರ, ಬಿಟ್);
                        ಆದರೆ (tile_src[byte_ಸ್ಥಾನಾಂತರ] & (1u << ಬಿಟ್)) {
                            ಖಂಡ.qs[k / 8] |= (uint8_t) (1u << (k % 8));
                        }
                    }
                }
            }
        }
    }
}

// --- Top-level Repack / Unpack Dispatchers ---

ಶೂನ್ಯ htp_repack_tensor_tiled(ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t size) {
    ಆಯ್ಕೆ (tensor->type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
            repack_q1_0_tiled(tensor, ದತ್ತಾಂಶ, 0, size);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
            repack_q4_0_tiled(tensor, ದತ್ತಾಂಶ, 0, size);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
            repack_q4_1_tiled(tensor, ದತ್ತಾಂಶ, 0, size);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            repack_q8_0_tiled(tensor, ದತ್ತಾಂಶ, 0, size);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
            repack_mxfp4_tiled(tensor, ದತ್ತಾಂಶ, 0, size);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
            repack_q2_0_tiled(tensor, ದತ್ತಾಂಶ, 0, size);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
            repack_q2_K_tiled(tensor, ದತ್ತಾಂಶ, 0, size);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
            repack_q3_K_tiled(tensor, ದತ್ತಾಂಶ, 0, size);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
            repack_q4_K_tiled(tensor, ದತ್ತಾಂಶ, 0, size);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
            repack_q5_K_tiled(tensor, ದತ್ತಾಂಶ, 0, size);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
            repack_q6_K_tiled(tensor, ದತ್ತಾಂಶ, 0, size);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            repack_float_ಗಮ್ಯ_f16_tiled<uint16_t>(tensor, ದತ್ತಾಂಶ, 0, size,
                [](uint16_t v) { ಹಿಂತಿರುಗಿ v; });
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            repack_float_ಗಮ್ಯ_f16_tiled<ದಶಮಾಂಶ>(tensor, ದತ್ತಾಂಶ, 0, size,
                [](ದಶಮಾಂಶ v) { ಹಿಂತಿರುಗಿ (uint16_t) GGML_FP32_ಗಮ್ಯ_FP16(v); });
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            repack_float_ಗಮ್ಯ_f16_tiled<ggml_bf16_t>(tensor, ದತ್ತಾಂಶ, 0, size,
                [](ggml_bf16_t v) { ಹಿಂತಿರುಗಿ (uint16_t) GGML_FP32_ಗಮ್ಯ_FP16(GGML_BF16_ಗಮ್ಯ_FP32(v)); });
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ನಿಲ್ಲಿಸು;
    }
}

ಶೂನ್ಯ htp_unpack_tensor_tiled(ಶೂನ್ಯ * ದತ್ತಾಂಶ, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    GGML_ASSERT(ಸ್ಥಾನಾಂತರ + size <= ggml_nbytes(tensor));

    ಆಯ್ಕೆ (tensor->type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
            repack_tiled_q1_0(ದತ್ತಾಂಶ, tensor, ಸ್ಥಾನಾಂತರ, size);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
            repack_tiled_q4_0(ದತ್ತಾಂಶ, tensor, ಸ್ಥಾನಾಂತರ, size);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
            repack_tiled_q4_1(ದತ್ತಾಂಶ, tensor, ಸ್ಥಾನಾಂತರ, size);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            repack_tiled_q8_0(ದತ್ತಾಂಶ, tensor, ಸ್ಥಾನಾಂತರ, size);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
            repack_tiled_mxfp4(ದತ್ತಾಂಶ, tensor, ಸ್ಥಾನಾಂತರ, size);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
            repack_tiled_q2_0(ದತ್ತಾಂಶ, tensor, ಸ್ಥಾನಾಂತರ, size);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
            repack_tiled_q2_K(ದತ್ತಾಂಶ, tensor, ಸ್ಥಾನಾಂತರ, size);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
            repack_tiled_q3_K(ದತ್ತಾಂಶ, tensor, ಸ್ಥಾನಾಂತರ, size);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
            repack_tiled_q4_K(ದತ್ತಾಂಶ, tensor, ಸ್ಥಾನಾಂತರ, size);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
            repack_tiled_q5_K(ದತ್ತಾಂಶ, tensor, ಸ್ಥಾನಾಂತರ, size);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
            repack_tiled_q6_K(ದತ್ತಾಂಶ, tensor, ಸ್ಥಾನಾಂತರ, size);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            unpack_f16_tiled_ಗಮ್ಯ_float<uint16_t>(ದತ್ತಾಂಶ, tensor, ಸ್ಥಾನಾಂತರ, size,
                [](uint16_t v) { ಹಿಂತಿರುಗಿ v; });
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            unpack_f16_tiled_ಗಮ್ಯ_float<ದಶಮಾಂಶ>(ದತ್ತಾಂಶ, tensor, ಸ್ಥಾನಾಂತರ, size,
                [](uint16_t v) { ಹಿಂತಿರುಗಿ ggml_fp16_ಗಮ್ಯ_fp32((ggml_fp16_t) v); });
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            unpack_f16_tiled_ಗಮ್ಯ_float<ggml_bf16_t>(ದತ್ತಾಂಶ, tensor, ಸ್ಥಾನಾಂತರ, size,
                [](uint16_t v) { ಹಿಂತಿರುಗಿ ggml_fp32_ಗಮ್ಯ_bf16(ggml_fp16_ಗಮ್ಯ_fp32((ggml_fp16_t) v)); });
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಸ್ಮರಣೆ_ನಕಲು(ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, size);
            ನಿಲ್ಲಿಸು;
    }
}
