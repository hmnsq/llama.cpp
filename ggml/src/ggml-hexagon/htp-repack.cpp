#ਅਨ੍ਤਰ੍ਭਾਵ "htp-repack.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_IMPL_CPP
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-quants.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਅਨ੍ਤਰ੍ਭਾਵ <initializer_list>
#ਅਨ੍ਤਰ੍ਭਾਵ <thread>

#ਜੇ defined(__ARM_NEON) || defined(__aarch64__)
#    include <arm_neon.h>
#ਜੇ_ਅੰਤ

// --- 4-bit block-32 nibble helpers (shared by Q4_0, IQ4_NL, Q4_1, MXFP4) ---

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ unpack_nibbles_32(uint8_t * dst, ਅਚਲ uint8_t * qs) {
    ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i = 0; i < 16; ++i) {
        dst[i + 0]  = qs[i] & 0x0F;
        dst[i + 16] = qs[i] >> 4;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ pack_nibbles_32(uint8_t * qs, ਅਚਲ uint8_t * src) {
    ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i = 0; i < 16; ++i) {
        qs[i] = src[i + 0] | (src[i + 16] << 4);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ transpose_nibbles_ਮੰਜ਼ਿਲ_tile(uint8_t * tile_dst, ਅਚਲ uint8_t tile_quants[32][32]) {
    ਲਈ (ਪੂਰਨ cp = 0; cp < 16; cp++) {
        ਲਈ (ਪੂਰਨ ਪੰਕਤੀ = 0; ਪੰਕਤੀ < 32; ਪੰਕਤੀ++) {
            tile_dst[cp * 32 + ਪੰਕਤੀ] = (tile_quants[ਪੰਕਤੀ][2 * cp + 1] << 4) | tile_quants[ਪੰਕਤੀ][2 * cp];
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ transpose_tile_ਮੰਜ਼ਿਲ_nibbles(uint8_t tile_quants[32][32], ਅਚਲ uint8_t * tile_src) {
    ਲਈ (ਪੂਰਨ cp = 0; cp < 16; cp++) {
        ਲਈ (ਪੂਰਨ ਪੰਕਤੀ = 0; ਪੰਕਤੀ < 32; ਪੰਕਤੀ++) {
            ਅਚਲ uint8_t val = tile_src[cp * 32 + ਪੰਕਤੀ];
            tile_quants[ਪੰਕਤੀ][2 * cp + 0] = val & 0x0F;
            tile_quants[ਪੰਕਤੀ][2 * cp + 1] = val >> 4;
        }
    }
}

#ਜੇ defined(__ARM_NEON) || defined(__aarch64__)
ਸਥਿਰ ਖਾਲੀ repack_neon_transpose8x8(ਅਚਲ uint8x8_t ਪੰਕਤੀਆਂ[8], uint8x8_t columns[8]) {
    ਅਚਲ uint8x8x2_t t0 = vtrn_u8(ਪੰਕਤੀਆਂ[0], ਪੰਕਤੀਆਂ[1]);
    ਅਚਲ uint8x8x2_t t1 = vtrn_u8(ਪੰਕਤੀਆਂ[2], ਪੰਕਤੀਆਂ[3]);
    ਅਚਲ uint8x8x2_t t2 = vtrn_u8(ਪੰਕਤੀਆਂ[4], ਪੰਕਤੀਆਂ[5]);
    ਅਚਲ uint8x8x2_t t3 = vtrn_u8(ਪੰਕਤੀਆਂ[6], ਪੰਕਤੀਆਂ[7]);
    ਅਚਲ uint16x4x2_t t4 = vtrn_u16(vreinterpret_u16_u8(t0.val[0]), vreinterpret_u16_u8(t1.val[0]));
    ਅਚਲ uint16x4x2_t t5 = vtrn_u16(vreinterpret_u16_u8(t0.val[1]), vreinterpret_u16_u8(t1.val[1]));
    ਅਚਲ uint16x4x2_t t6 = vtrn_u16(vreinterpret_u16_u8(t2.val[0]), vreinterpret_u16_u8(t3.val[0]));
    ਅਚਲ uint16x4x2_t t7 = vtrn_u16(vreinterpret_u16_u8(t2.val[1]), vreinterpret_u16_u8(t3.val[1]));
    ਅਚਲ uint32x2x2_t t8 = vtrn_u32(vreinterpret_u32_u16(t4.val[0]), vreinterpret_u32_u16(t6.val[0]));
    ਅਚਲ uint32x2x2_t t9 = vtrn_u32(vreinterpret_u32_u16(t4.val[1]), vreinterpret_u32_u16(t6.val[1]));
    ਅਚਲ uint32x2x2_t ta = vtrn_u32(vreinterpret_u32_u16(t5.val[0]), vreinterpret_u32_u16(t7.val[0]));
    ਅਚਲ uint32x2x2_t tb = vtrn_u32(vreinterpret_u32_u16(t5.val[1]), vreinterpret_u32_u16(t7.val[1]));
    columns[0] = vreinterpret_u8_u32(t8.val[0]);
    columns[1] = vreinterpret_u8_u32(ta.val[0]);
    columns[2] = vreinterpret_u8_u32(t9.val[0]);
    columns[3] = vreinterpret_u8_u32(tb.val[0]);
    columns[4] = vreinterpret_u8_u32(t8.val[1]);
    columns[5] = vreinterpret_u8_u32(ta.val[1]);
    columns[6] = vreinterpret_u8_u32(t9.val[1]);
    columns[7] = vreinterpret_u8_u32(tb.val[1]);
}

ਸਥਿਰ uint8x8_t repack_neon_pack_4bit_ਜੋੜੇ(uint8x8_t ਪਹਿਲਾ, uint8x8_t second, ਤਾਰਕਿਕ ਉੱਚਾ) {
    ਅਚਲ uint8x8x2_t separated = vuzp_u8(ਪਹਿਲਾ, second);
    ਅਚਲ uint8x8_t ਨਕਾਬ = vdup_n_u8(0x0F);
    uint8x8_t even = separated.val[0];
    uint8x8_t odd  = separated.val[1];

    ਜੇ (ਉੱਚਾ) {
        even = vshr_n_u8(even, 4);
        odd  = vshr_n_u8(odd, 4);
    }

    even = vand_u8(even, ਨਕਾਬ);
    odd  = vand_u8(odd, ਨਕਾਬ);
    ਵਾਪਸ vorr_u8(vshl_n_u8(odd, 4), even);
}

ਸਥਿਰ ਖਾਲੀ repack_neon_4bit_ਪੰਕਤੀਆਂ(ਅਚਲ uint8_t * qs[8], uint8_t * tile_dst, ਪੂਰਨ row0) {
    uint8x8_t ਨੀਵਾਂ_ਪੰਕਤੀਆਂ[8];
    uint8x8_t ਉੱਚਾ_ਪੰਕਤੀਆਂ[8];
    uint8x8_t columns[8];

    ਲਈ (ਪੂਰਨ rr = 0; rr < 8; ++rr) {
        ਅਚਲ uint8x16_t packed = vld1q_u8(qs[rr]);
        ਅਚਲ uint8x8_t ਪਹਿਲਾ = vget_ਨੀਵਾਂ_u8(packed);
        ਅਚਲ uint8x8_t second = vget_ਉੱਚਾ_u8(packed);
        ਨੀਵਾਂ_ਪੰਕਤੀਆਂ[rr]  = repack_neon_pack_4bit_ਜੋੜੇ(ਪਹਿਲਾ, second, ਝੂਠ);
        ਉੱਚਾ_ਪੰਕਤੀਆਂ[rr] = repack_neon_pack_4bit_ਜੋੜੇ(ਪਹਿਲਾ, second, ਸੱਚ);
    }

    repack_neon_transpose8x8(ਨੀਵਾਂ_ਪੰਕਤੀਆਂ, columns);
    ਲਈ (ਪੂਰਨ cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + cp * 32 + row0, columns[cp]);
    }

    repack_neon_transpose8x8(ਉੱਚਾ_ਪੰਕਤੀਆਂ, columns);
    ਲਈ (ਪੂਰਨ cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + (cp + 8) * 32 + row0, columns[cp]);
    }
}

ਸਥਿਰ ਖਾਲੀ repack_neon_k4_ਪੰਕਤੀਆਂ(ਅਚਲ uint8_t * qs[8], uint8_t * tile_dst, ਪੂਰਨ row0, ਤਾਰਕਿਕ ਉੱਚਾ) {
    uint8x8_t ਪਹਿਲਾ_ਪੰਕਤੀਆਂ[8];
    uint8x8_t ਦੂਜਾ_ਪੰਕਤੀਆਂ[8];
    uint8x8_t columns[8];

    ਲਈ (ਪੂਰਨ rr = 0; rr < 8; ++rr) {
        ਪਹਿਲਾ_ਪੰਕਤੀਆਂ[rr] = repack_neon_pack_4bit_ਜੋੜੇ(vld1_u8(qs[rr]), vld1_u8(qs[rr] + 8), ਉੱਚਾ);
        ਦੂਜਾ_ਪੰਕਤੀਆਂ[rr] = repack_neon_pack_4bit_ਜੋੜੇ(vld1_u8(qs[rr] + 16), vld1_u8(qs[rr] + 24), ਉੱਚਾ);
    }

    repack_neon_transpose8x8(ਪਹਿਲਾ_ਪੰਕਤੀਆਂ, columns);
    ਲਈ (ਪੂਰਨ cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + cp * 32 + row0, columns[cp]);
    }

    repack_neon_transpose8x8(ਦੂਜਾ_ਪੰਕਤੀਆਂ, columns);
    ਲਈ (ਪੂਰਨ cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + (cp + 8) * 32 + row0, columns[cp]);
    }
}

ਸਥਿਰ ਖਾਲੀ repack_neon_q8_ਪੰਕਤੀਆਂ(ਅਚਲ uint8_t * qs[8], uint8_t * tile_dst, ਪੂਰਨ row0) {
    uint8x8_t ਪੰਕਤੀਆਂ[8];
    uint8x8_t columns[8];

    ਲਈ (ਪੂਰਨ cp = 0; cp < 16; cp += 4) {
        ਲਈ (ਪੂਰਨ rr = 0; rr < 8; ++rr) {
            ਪੰਕਤੀਆਂ[rr] = vld1_u8(qs[rr] + 2 * cp);
        }
        repack_neon_transpose8x8(ਪੰਕਤੀਆਂ, columns);

        ਲਈ (ਪੂਰਨ pair = 0; pair < 4; ++pair) {
            ਅਚਲ uint8x8x2_t interleaved = vzip_u8(columns[2 * pair], columns[2 * pair + 1]);
            uint8_t * dst = tile_dst + (cp + pair) * 64 + 2 * row0;
            vst1_u8(dst, interleaved.val[0]);
            vst1_u8(dst + 8, interleaved.val[1]);
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ repack_store_u32(uint8_t * dst, uint32_t v) {
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst, &v, ਆਕਾਰ(v));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint32x4_t repack_neon_transpose_2bit_4x4(uint32x4_t w) {
    uint32x4_t t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 6)), vdupq_n_u32(0x00CC00CCu));
    w = veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 6)));
    t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 12)), vdupq_n_u32(0x0000F0F0u));
    ਵਾਪਸ veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 12)));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ repack_neon_q2_0_ਖੰਡ(ਅਚਲ uint8_t * qs, uint8_t * tile, ਪੂਰਨ ਪੰਕਤੀ) {
    ਅਚਲ uint32x4_t w = repack_neon_transpose_2bit_4x4(vreinterpretq_u32_u8(vld1q_u8(qs)));
    repack_store_u32(tile + 0 * 128 + 4 * ਪੰਕਤੀ, vgetq_lane_u32(w, 0));
    repack_store_u32(tile + 1 * 128 + 4 * ਪੰਕਤੀ, vgetq_lane_u32(w, 1));
    repack_store_u32(tile + 2 * 128 + 4 * ਪੰਕਤੀ, vgetq_lane_u32(w, 2));
    repack_store_u32(tile + 3 * 128 + 4 * ਪੰਕਤੀ, vgetq_lane_u32(w, 3));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ repack_neon_q1_0_ਖੰਡ(ਅਚਲ uint8_t * qs, uint8_t * tile, ਪੂਰਨ ਪੰਕਤੀ) {
    uint32x4_t w = vreinterpretq_u32_u8(vld1q_u8(qs));
    uint32x4_t t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 1)), vdupq_n_u32(0x22222222u));
    w = veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 1)));
    t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 3)), vdupq_n_u32(0x0A0A0A0Au));
    w = veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 3)));
    w = repack_neon_transpose_2bit_4x4(w);
    repack_store_u32(tile + 0 * 128 + 4 * ਪੰਕਤੀ, vgetq_lane_u32(w, 0));
    repack_store_u32(tile + 1 * 128 + 4 * ਪੰਕਤੀ, vgetq_lane_u32(w, 1));
    repack_store_u32(tile + 2 * 128 + 4 * ਪੰਕਤੀ, vgetq_lane_u32(w, 2));
    repack_store_u32(tile + 3 * 128 + 4 * ਪੰਕਤੀ, vgetq_lane_u32(w, 3));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ repack_neon_pack_2bit_plane(uint8x16_t v0, uint8x16_t v1, uint8_t * dst, ਪੂਰਨ ਪੰਕਤੀ) {
    ਅਚਲ uint32x4x2_t z   = vuzpq_u32(vreinterpretq_u32_u8(v0), vreinterpretq_u32_u8(v1));
    ਅਚਲ uint32x4_t   p01 = vorrq_u32(z.val[0], vshlq_n_u32(z.val[1], 2));
    ਅਚਲ uint32x2x2_t u   = vuzp_u32(vget_ਨੀਵਾਂ_u32(p01), vget_ਉੱਚਾ_u32(p01));
    ਅਚਲ uint32x2_t   ਬਾਹਰਲਾ = vorr_u32(u.val[0], vshl_n_u32(u.val[1], 4));
    repack_store_u32(dst + 0 * 128 + ਪੰਕਤੀ * 4, vget_lane_u32(ਬਾਹਰਲਾ, 0));
    repack_store_u32(dst + 1 * 128 + ਪੰਕਤੀ * 4, vget_lane_u32(ਬਾਹਰਲਾ, 1));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ repack_neon_q2_3_k_low2(ਅਚਲ uint8_t * qs, ਪੂਰਨ kt_local, uint8_t * lo_pl, ਪੂਰਨ ਪੰਕਤੀ) {
    ਅਚਲ uint8_t *  qs_sub = qs + (kt_local / 4) * 32;
    ਅਚਲ int8x16_t  ਖਿਸਕਾਓ  = vdupq_n_s8(-2 * (kt_local % 4));
    ਅਚਲ uint8x16_t ਨਕਾਬ   = vdupq_n_u8(0x03);
    ਅਚਲ uint8x16_t v0     = vandq_u8(vshlq_u8(vld1q_u8(qs_sub +  0), ਖਿਸਕਾਓ), ਨਕਾਬ);
    ਅਚਲ uint8x16_t v1     = vandq_u8(vshlq_u8(vld1q_u8(qs_sub + 16), ਖਿਸਕਾਓ), ਨਕਾਬ);
    repack_neon_pack_2bit_plane(v0, v1, lo_pl, ਪੰਕਤੀ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ repack_neon_q3_k_hmask(ਅਚਲ uint8_t * hmask, ਪੂਰਨ kt_local, uint8_t * neg_pl, ਪੂਰਨ ਪੰਕਤੀ) {
    ਸਥਿਰ ਅਚਲ uint32_t ਨਕਾਬ_ਸ਼ਬਦ[4] = { 0x01010101u, 0x02020202u, 0x04040404u, 0x08080808u };
    ਅਚਲ uint8x16_t m0   = vreinterpretq_u8_u32(vld1q_u32(ਨਕਾਬ_ਸ਼ਬਦ));
    ਅਚਲ uint8x16_t m1   = vshlq_n_u8(m0, 4);
    ਅਚਲ uint8x16_t hbit = vdupq_n_u8((uint8_t) (1u << kt_local));
    ਅਚਲ uint8x16_t b0   = vbicq_u8(m0, vtstq_u8(vld1q_u8(hmask +  0), hbit));
    ਅਚਲ uint8x16_t b1   = vbicq_u8(m1, vtstq_u8(vld1q_u8(hmask + 16), hbit));
    ਅਚਲ uint32x4_t b01  = vreinterpretq_u32_u8(vorrq_u8(b0, b1));
    ਅਚਲ uint32x2_t f0   = vorr_u32(vget_ਨੀਵਾਂ_u32(b01), vget_ਉੱਚਾ_u32(b01));
    ਅਚਲ uint32x2_t f1   = vpadd_u32(f0, f0);
    repack_store_u32(neg_pl + ਪੰਕਤੀ * 4, vget_lane_u32(f1, 0));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ repack_neon_q6_k(ਅਚਲ ਖੰਡ_q6_K * b, ਪੂਰਨ kt_local, uint8_t * lo_pl, uint8_t * hi_pl, ਪੂਰਨ ਪੰਕਤੀ) {
    ਅਚਲ ਪੂਰਨ        c      = kt_local / 4;
    ਅਚਲ ਪੂਰਨ        g_sb   = kt_local % 4;
    ਅਚਲ uint8_t *  ql     = b->ql + c * 64 + (g_sb & 1) * 32;
    ਅਚਲ uint8_t *  qh     = b->qh + c * 32;
    ਅਚਲ int8x16_t  lshift = vdupq_n_s8((g_sb >= 2) ? -4 : 0);
    ਅਚਲ int8x16_t  hshift = vdupq_n_s8(-2 * g_sb);
    ਅਚਲ uint8x16_t mask4  = vdupq_n_u8(0x0F);
    ਅਚਲ uint8x16_t mask2  = vdupq_n_u8(0x03);

    ਅਚਲ uint8x16_t lo0  = vandq_u8(vshlq_u8(vld1q_u8(ql +  0), lshift), mask4);
    ਅਚਲ uint8x16_t lo1  = vandq_u8(vshlq_u8(vld1q_u8(ql + 16), lshift), mask4);
    ਅਚਲ uint32x4x2_t lz = vuzpq_u32(vreinterpretq_u32_u8(lo0), vreinterpretq_u32_u8(lo1));
    ਅਚਲ uint32x4_t lout = vorrq_u32(lz.val[0], vshlq_n_u32(lz.val[1], 4));
    repack_store_u32(lo_pl + 0 * 128 + ਪੰਕਤੀ * 4, vgetq_lane_u32(lout, 0));
    repack_store_u32(lo_pl + 1 * 128 + ਪੰਕਤੀ * 4, vgetq_lane_u32(lout, 1));
    repack_store_u32(lo_pl + 2 * 128 + ਪੰਕਤੀ * 4, vgetq_lane_u32(lout, 2));
    repack_store_u32(lo_pl + 3 * 128 + ਪੰਕਤੀ * 4, vgetq_lane_u32(lout, 3));

    ਅਚਲ uint8x16_t hi0 = vandq_u8(vshlq_u8(vld1q_u8(qh +  0), hshift), mask2);
    ਅਚਲ uint8x16_t hi1 = vandq_u8(vshlq_u8(vld1q_u8(qh + 16), hshift), mask2);
    repack_neon_pack_2bit_plane(hi0, hi1, hi_pl, ਪੰਕਤੀ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ repack_neon_q5_k_qh(ਅਚਲ uint8_t * qh, uint8_t hbit, uint8_t * plane, ਪੂਰਨ ਪੰਕਤੀ) {
    ਸਥਿਰ ਅਚਲ uint8_t ਨਕਾਬ_ਬਾਈਟਾਂ[16] = { 1, 2, 1, 2, 1, 2, 1, 2, 4, 8, 4, 8, 4, 8, 4, 8 };
    ਅਚਲ uint8x16_t m0   = vld1q_u8(ਨਕਾਬ_ਬਾਈਟਾਂ);
    ਅਚਲ uint8x16_t m1   = vshlq_n_u8(m0, 4);
    ਅਚਲ uint8x16_t vbit = vdupq_n_u8(hbit);
    ਅਚਲ uint8x16_t b0   = vandq_u8(vtstq_u8(vld1q_u8(qh +  0), vbit), m0);
    ਅਚਲ uint8x16_t b1   = vandq_u8(vtstq_u8(vld1q_u8(qh + 16), vbit), m1);
    ਅਚਲ uint8x16_t b01  = vorrq_u8(b0, b1);
    ਅਚਲ uint8x8_t  fold = vorr_u8(vget_ਨੀਵਾਂ_u8(b01), vget_ਉੱਚਾ_u8(b01));
    ਅਚਲ uint8x8_t  ਬਾਹਰਲਾ  = vpadd_u8(fold, fold);
    plane[0 * 32 + ਪੰਕਤੀ] = vget_lane_u8(ਬਾਹਰਲਾ, 0);
    plane[1 * 32 + ਪੰਕਤੀ] = vget_lane_u8(ਬਾਹਰਲਾ, 1);
    plane[2 * 32 + ਪੰਕਤੀ] = vget_lane_u8(ਬਾਹਰਲਾ, 2);
    plane[3 * 32 + ਪੰਕਤੀ] = vget_lane_u8(ਬਾਹਰਲਾ, 3);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint16x8_t repack_neon_ਲੱਦੋ_f16x8(ਅਚਲ uint16_t * src) {
    ਵਾਪਸ vld1q_u16(src);
}

#ਜੇ defined(__aarch64__) || defined(__ARM_FP16_FORMAT_IEEE)
ਸਥਿਰ ਸਤਰਅੰਦਰ uint16x8_t repack_neon_ਲੱਦੋ_f16x8(ਅਚਲ ਦਸ਼ਮਲਵ * src) {
    ਅਚਲ float16x4_t lo = vcvt_f16_f32(vld1q_f32(src));
    ਅਚਲ float16x4_t hi = vcvt_f16_f32(vld1q_f32(src + 4));
    ਵਾਪਸ vreinterpretq_u16_f16(vcombine_f16(lo, hi));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint16x8_t repack_neon_ਲੱਦੋ_f16x8(ਅਚਲ ggml_bf16_t * src) {
    ਅਚਲ uint16x8_t  bf   = vld1q_u16((ਅਚਲ uint16_t *) src);
    ਅਚਲ float32x4_t f_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_ਨੀਵਾਂ_u16(bf), 16));
    ਅਚਲ float32x4_t f_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_ਉੱਚਾ_u16(bf), 16));
    ਅਚਲ float16x4_t lo   = vcvt_f16_f32(f_lo);
    ਅਚਲ float16x4_t hi   = vcvt_f16_f32(f_hi);
    ਵਾਪਸ vreinterpretq_u16_f16(vcombine_f16(lo, hi));
}
#ਨਹੀਂਤਾਂ
ਸਥਿਰ ਸਤਰਅੰਦਰ uint16x8_t repack_neon_ਲੱਦੋ_f16x8(ਅਚਲ ਦਸ਼ਮਲਵ * src) {
    uint16_t tmp[8];
    ਲਈ (ਪੂਰਨ i = 0; i < 8; ++i) {
        tmp[i] = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(src[i]);
    }
    ਵਾਪਸ vld1q_u16(tmp);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint16x8_t repack_neon_ਲੱਦੋ_f16x8(ਅਚਲ ggml_bf16_t * src) {
    uint16_t tmp[8];
    ਲਈ (ਪੂਰਨ i = 0; i < 8; ++i) {
        tmp[i] = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(GGML_BF16_ਮੰਜ਼ਿਲ_FP32(src[i]));
    }
    ਵਾਪਸ vld1q_u16(tmp);
}
#ਜੇ_ਅੰਤ

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ SrcT>
ਸਥਿਰ ਖਾਲੀ repack_neon_f16_tile(uint16_t * tile_dst, ਅਚਲ SrcT * src_ਟੁਕੜਾ, int64_t ਪੰਕਤੀ_stride, ਪੂਰਨ ct, ਪੂਰਨ kt) {
    ਲਈ (ਪੂਰਨ row0 = 0; row0 < 32; row0 += 4) {
        ਅਚਲ SrcT * r0 = src_ਟੁਕੜਾ + (int64_t) (ct * 32 + row0 + 0) * ਪੰਕਤੀ_stride + kt * 32;
        ਅਚਲ SrcT * r1 = src_ਟੁਕੜਾ + (int64_t) (ct * 32 + row0 + 1) * ਪੰਕਤੀ_stride + kt * 32;
        ਅਚਲ SrcT * r2 = src_ਟੁਕੜਾ + (int64_t) (ct * 32 + row0 + 2) * ਪੰਕਤੀ_stride + kt * 32;
        ਅਚਲ SrcT * r3 = src_ਟੁਕੜਾ + (int64_t) (ct * 32 + row0 + 3) * ਪੰਕਤੀ_stride + kt * 32;
        ਲਈ (ਪੂਰਨ cp = 0; cp < 16; cp += 4) {
            ਅਚਲ uint32x4_t v0 = vreinterpretq_u32_u16(repack_neon_ਲੱਦੋ_f16x8(r0 + 2 * cp));
            ਅਚਲ uint32x4_t v1 = vreinterpretq_u32_u16(repack_neon_ਲੱਦੋ_f16x8(r1 + 2 * cp));
            ਅਚਲ uint32x4_t v2 = vreinterpretq_u32_u16(repack_neon_ਲੱਦੋ_f16x8(r2 + 2 * cp));
            ਅਚਲ uint32x4_t v3 = vreinterpretq_u32_u16(repack_neon_ਲੱਦੋ_f16x8(r3 + 2 * cp));

            ਅਚਲ uint32x2x2_t t0 = vtrn_u32(vget_ਨੀਵਾਂ_u32(v0),  vget_ਨੀਵਾਂ_u32(v1));
            ਅਚਲ uint32x2x2_t t1 = vtrn_u32(vget_ਉੱਚਾ_u32(v0), vget_ਉੱਚਾ_u32(v1));
            ਅਚਲ uint32x2x2_t t2 = vtrn_u32(vget_ਨੀਵਾਂ_u32(v2),  vget_ਨੀਵਾਂ_u32(v3));
            ਅਚਲ uint32x2x2_t t3 = vtrn_u32(vget_ਉੱਚਾ_u32(v2), vget_ਉੱਚਾ_u32(v3));

            vst1q_u16(tile_dst + (cp + 0) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t0.val[0], t2.val[0])));
            vst1q_u16(tile_dst + (cp + 1) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t0.val[1], t2.val[1])));
            vst1q_u16(tile_dst + (cp + 2) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t1.val[0], t3.val[0])));
            vst1q_u16(tile_dst + (cp + 3) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t1.val[1], t3.val[1])));
        }
    }
}
#ਜੇ_ਅੰਤ

// --- Generic 32x32 tile grid walkers for block-32 and float types ---

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ Fn>
ਸਥਿਰ ਖਾਲੀ repack_parallel_for_ct(size_t ਟੁਕੜਾ_ਬਾਈਟਾਂ, ਪੂਰਨ n_ਖਾਨਾ_tiles, Fn && fn) {
    ਪੂਰਨ n_ਧਾਗੇ = 1;
    ਜੇ (ਟੁਕੜਾ_ਬਾਈਟਾਂ > 128 * 1024) {
        ਲਈ (ਪੂਰਨ nt : { 4, 3, 2 }) {
            ਜੇ (n_ਖਾਨਾ_tiles % nt == 0) {
                n_ਧਾਗੇ = nt;
                ਰੋਕੋ;
            }
        }
    }

    ਜੇ (n_ਧਾਗੇ <= 1) {
        fn(0, n_ਖਾਨਾ_tiles);
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ chunk = n_ਖਾਨਾ_tiles / n_ਧਾਗੇ;
    std::thread workers[3];
    ਲਈ (ਪੂਰਨ i = 0; i < n_ਧਾਗੇ - 1; ++i) {
        workers[i] = std::thread([&fn, i, chunk]() {
            fn(i * chunk, (i + 1) * chunk);
        });
    }
    fn((n_ਧਾਗੇ - 1) * chunk, n_ਖਾਨਾ_tiles);
    ਲਈ (ਪੂਰਨ i = 0; i < n_ਧਾਗੇ - 1; ++i) {
        workers[i].join();
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ SrcElemT, ਪੂਰਨ ELEMS_ਪ੍ਰਤੀ_KTILE, ਕਿਸਮ_ਨਾਂ PackTileFn>
ਸਥਿਰ ਖਾਲੀ repack_ਮੰਜ਼ਿਲ_tiled_grid_32(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size,
                                    size_t tile_ਮਾਪ, PackTileFn && pack_tile) {
    ਅਚਲ SrcElemT * src_ਮੈਟ੍ਰਿਕਸ = (ਅਚਲ SrcElemT *) ਅੰਕੜੇ;
    ਅਚਲ int64_t ne0 = t->ne[0];
    ਅਚਲ int64_t ne1 = t->ne[1];
    ਅਚਲ int64_t ne2 = t->ne[2];
    ਅਚਲ int64_t ne3 = t->ne[3];
    ਅਚਲ ਪੂਰਨ n_ਖਾਨਾ_tiles = hex_ਗੇੜਾ_ਉੱਪਰ(ne1, 32) / 32;
    ਅਚਲ ਪੂਰਨ n_k_tiles   = hex_ਗੇੜਾ_ਉੱਪਰ(ne0, 32) / 32;
    ਅਚਲ size_t ਮੈਟ੍ਰਿਕਸ_ਮਾਪ = (size_t) n_ਖਾਨਾ_tiles * n_k_tiles * tile_ਮਾਪ;

    ਅਚਲ size_t ਟੁਕੜਾ_ਮਾਪ = (size_t) ne1 * ggml_ਪੰਕਤੀ_ਮਾਪ(t->type, ne0);
    ਅਚਲ int64_t ਸ਼ੁਰੂ_ਟੁਕੜਾ = ਵਿੱਥ / ਟੁਕੜਾ_ਮਾਪ;
    int64_t ਸਿਰਾ_ਟੁਕੜਾ = (ਵਿੱਥ + size + ਟੁਕੜਾ_ਮਾਪ - 1) / ਟੁਕੜਾ_ਮਾਪ;
    ਜੇ (ਸਿਰਾ_ਟੁਕੜਾ > ne2 * ne3) {
        ਸਿਰਾ_ਟੁਕੜਾ = ne2 * ne3;
    }

    ਅਚਲ int64_t ਪੰਕਤੀ_stride = (ne0 / (32 / ELEMS_ਪ੍ਰਤੀ_KTILE));

    ਲਈ (int64_t ਟੁਕੜਾ_idx = ਸ਼ੁਰੂ_ਟੁਕੜਾ; ਟੁਕੜਾ_idx < ਸਿਰਾ_ਟੁਕੜਾ; ਟੁਕੜਾ_idx++) {
        ਅਚਲ SrcElemT * src_ਟੁਕੜਾ = src_ਮੈਟ੍ਰਿਕਸ + (ਟੁਕੜਾ_idx - ਸ਼ੁਰੂ_ਟੁਕੜਾ) * (ne1 * ਪੰਕਤੀ_stride);
        uint8_t * ਮੈਟ੍ਰਿਕਸ_dst = (uint8_t *) t->ਅੰਕੜੇ + ਟੁਕੜਾ_idx * ਮੈਟ੍ਰਿਕਸ_ਮਾਪ;

        repack_parallel_for_ct(ਮੈਟ੍ਰਿਕਸ_ਮਾਪ, n_ਖਾਨਾ_tiles, [&](ਪੂਰਨ ct0, ਪੂਰਨ ct1) {
            ਲਈ (ਪੂਰਨ ct = ct0; ct < ct1; ct++) {
                ਲਈ (ਪੂਰਨ kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = ਮੈਟ੍ਰਿਕਸ_dst + ((size_t) ct * n_k_tiles + kt) * tile_ਮਾਪ;
                    pack_tile(tile_dst, src_ਟੁਕੜਾ, ne0, ne1, ਪੰਕਤੀ_stride, ct, kt);
                }
            }
        });
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ DstElemT, ਪੂਰਨ ELEMS_ਪ੍ਰਤੀ_KTILE, ਕਿਸਮ_ਨਾਂ UnpackTileFn>
ਸਥਿਰ ਖਾਲੀ unpack_from_tiled_grid_32(ਖਾਲੀ * ਅੰਕੜੇ, ਅਚਲ ggml_tensor * t, size_t ਵਿੱਥ, size_t size,
                                      size_t tile_ਮਾਪ, UnpackTileFn && unpack_tile) {
    uint8_t * dst_ਬਾਈਟਾਂ = (uint8_t *) ਅੰਕੜੇ;
    ਅਚਲ int64_t ne0 = t->ne[0];
    ਅਚਲ int64_t ne1 = t->ne[1];
    ਅਚਲ int64_t ne2 = t->ne[2];
    ਅਚਲ int64_t ne3 = t->ne[3];
    ਅਚਲ ਪੂਰਨ n_ਖਾਨਾ_tiles = hex_ਗੇੜਾ_ਉੱਪਰ(ne1, 32) / 32;
    ਅਚਲ ਪੂਰਨ n_k_tiles   = hex_ਗੇੜਾ_ਉੱਪਰ(ne0, 32) / 32;
    ਅਚਲ size_t ਮੈਟ੍ਰਿਕਸ_ਮਾਪ = (size_t) n_ਖਾਨਾ_tiles * n_k_tiles * tile_ਮਾਪ;

    ਅਚਲ size_t ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ = ggml_ਪੰਕਤੀ_ਮਾਪ(t->type, ne0);
    ਅਚਲ size_t ਟੁਕੜਾ_ਮਾਪ     = (size_t) ne1 * ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ;
    ਅਚਲ int64_t ਸ਼ੁਰੂ_ਟੁਕੜਾ   = ਵਿੱਥ / ਟੁਕੜਾ_ਮਾਪ;
    int64_t ਸਿਰਾ_ਟੁਕੜਾ           = (ਵਿੱਥ + size + ਟੁਕੜਾ_ਮਾਪ - 1) / ਟੁਕੜਾ_ਮਾਪ;
    ਜੇ (ਸਿਰਾ_ਟੁਕੜਾ > ne2 * ne3) {
        ਸਿਰਾ_ਟੁਕੜਾ = ne2 * ne3;
    }

    ਅਚਲ int64_t ਪੰਕਤੀ_stride = (ne0 / (32 / ELEMS_ਪ੍ਰਤੀ_KTILE));

    ਲਈ (int64_t ਟੁਕੜਾ_idx = ਸ਼ੁਰੂ_ਟੁਕੜਾ; ਟੁਕੜਾ_idx < ਸਿਰਾ_ਟੁਕੜਾ; ਟੁਕੜਾ_idx++) {
        ਅਚਲ size_t cur_ਸ਼ੁਰੂ_byte     = (std::max)(ਵਿੱਥ, (size_t) ਟੁਕੜਾ_idx * ਟੁਕੜਾ_ਮਾਪ);
        ਅਚਲ size_t cur_ਸਿਰਾ_byte       = (std::min)(ਵਿੱਥ + size, (size_t) (ਟੁਕੜਾ_idx + 1) * ਟੁਕੜਾ_ਮਾਪ);
        ਅਚਲ size_t ਟੁਕੜਾ_ਵਿੱਥ_ਸ਼ੁਰੂ = cur_ਸ਼ੁਰੂ_byte - (size_t) ਟੁਕੜਾ_idx * ਟੁਕੜਾ_ਮਾਪ;
        ਅਚਲ size_t ਟੁਕੜਾ_ਵਿੱਥ_ਸਿਰਾ   = cur_ਸਿਰਾ_byte - (size_t) ਟੁਕੜਾ_idx * ਟੁਕੜਾ_ਮਾਪ;

        ਅਚਲ int64_t ਸ਼ੁਰੂ_ਪੰਕਤੀ = ਟੁਕੜਾ_ਵਿੱਥ_ਸ਼ੁਰੂ / ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ;
        ਅਚਲ int64_t ਸਿਰਾ_ਪੰਕਤੀ   = (std::min)((int64_t) ((ਟੁਕੜਾ_ਵਿੱਥ_ਸਿਰਾ + ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ - 1) / ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ), ne1);

        ਅਚਲ ਪੂਰਨ ਸ਼ੁਰੂ_ct = (ਪੂਰਨ) (ਸ਼ੁਰੂ_ਪੰਕਤੀ / 32);
        ਅਚਲ ਪੂਰਨ ਸਿਰਾ_ct   = (std::min)((ਪੂਰਨ) ((ਸਿਰਾ_ਪੰਕਤੀ + 31) / 32), n_ਖਾਨਾ_tiles);

        DstElemT * dst_ਟੁਕੜਾ = (DstElemT *) (dst_ਬਾਈਟਾਂ + (cur_ਸ਼ੁਰੂ_byte - ਵਿੱਥ));
        ਅਚਲ uint8_t * ਮੈਟ੍ਰਿਕਸ_src = (ਅਚਲ uint8_t *) t->ਅੰਕੜੇ + ਟੁਕੜਾ_idx * ਮੈਟ੍ਰਿਕਸ_ਮਾਪ;

        ਲਈ (ਪੂਰਨ ct = ਸ਼ੁਰੂ_ct; ct < ਸਿਰਾ_ct; ct++) {
            ਲਈ (ਪੂਰਨ kt = 0; kt < n_k_tiles; kt++) {
                ਅਚਲ uint8_t * tile_src = ਮੈਟ੍ਰਿਕਸ_src + ((size_t) ct * n_k_tiles + kt) * tile_ਮਾਪ;
                unpack_tile(dst_ਟੁਕੜਾ, tile_src, ne0, ਸ਼ੁਰੂ_ਪੰਕਤੀ, ਸਿਰਾ_ਪੰਕਤੀ, ਪੰਕਤੀ_stride, ct, kt);
            }
        }
    }
}

// --- Block-32 Quantized Types (Q4_0, IQ4_NL, Q4_1, Q8_0, MXFP4) ---

ਸਥਿਰ ਖਾਲੀ repack_q4_0_tiled(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    repack_ਮੰਜ਼ਿਲ_tiled_grid_32<ਖੰਡ_q4_0, 1>(t, ਅੰਕੜੇ, ਵਿੱਥ, size, HTP_TENSOR_TILE_ਮਾਪ_Q4_0,
        [](uint8_t * tile_dst, ਅਚਲ ਖੰਡ_q4_0 * src_ਟੁਕੜਾ, int64_t ne0, int64_t ne1, int64_t ਪੰਕਤੀ_stride, ਪੂਰਨ ct, ਪੂਰਨ kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
#ਜੇ defined(__ARM_NEON) || defined(__aarch64__)
            ਜੇ (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                ਲਈ (ਪੂਰਨ row0 = 0; row0 < 32; row0 += 8) {
                    ਅਚਲ uint8_t * qs[8];
                    ਲਈ (ਪੂਰਨ rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_ਟੁਕੜਾ[(ct * 32 + row0 + rr) * ਪੰਕਤੀ_stride + kt].qs;
                    }
                    repack_neon_4bit_ਪੰਕਤੀਆਂ(qs, tile_dst, row0);
                }

                ਲਈ (ਪੂਰਨ ਪੰਕਤੀ = 0; ਪੰਕਤੀ < 32; ++ਪੰਕਤੀ) {
                    scale_dst[ਪੰਕਤੀ] = src_ਟੁਕੜਾ[(ct * 32 + ਪੰਕਤੀ) * ਪੰਕਤੀ_stride + kt].d;
                }
                ਵਾਪਸ;
            }
#ਜੇ_ਅੰਤ
            uint8_t tile_quants[32][32];
            ਲਈ (ਪੂਰਨ ਪੰਕਤੀ = 0; ਪੰਕਤੀ < 32; ਪੰਕਤੀ++) {
                ਅਚਲ int64_t r = ct * 32 + ਪੰਕਤੀ;
                ਜੇ (r < ne1 && kt < ne0 / 32) {
                    ਅਚਲ ਖੰਡ_q4_0 & b = src_ਟੁਕੜਾ[r * ਪੰਕਤੀ_stride + kt];
                    unpack_nibbles_32(tile_quants[ਪੰਕਤੀ], b.qs);
                    scale_dst[ਪੰਕਤੀ] = b.d;
                } ਨਹੀਂਤਾਂ {
                    memset(tile_quants[ਪੰਕਤੀ], 8, 32);
                    scale_dst[ਪੰਕਤੀ] = 0;
                }
            }
            transpose_nibbles_ਮੰਜ਼ਿਲ_tile(tile_dst, tile_quants);
        });
}

ਸਥਿਰ ਖਾਲੀ repack_tiled_q4_0(ਖਾਲੀ * ਅੰਕੜੇ, ਅਚਲ ggml_tensor * t, size_t ਵਿੱਥ, size_t size) {
    unpack_from_tiled_grid_32<ਖੰਡ_q4_0, 1>(ਅੰਕੜੇ, t, ਵਿੱਥ, size, HTP_TENSOR_TILE_ਮਾਪ_Q4_0,
        [](ਖੰਡ_q4_0 * dst_ਟੁਕੜਾ, ਅਚਲ uint8_t * tile_src, int64_t ne0, int64_t ਸ਼ੁਰੂ_ਪੰਕਤੀ, int64_t ਸਿਰਾ_ਪੰਕਤੀ, int64_t ਪੰਕਤੀ_stride, ਪੂਰਨ ct, ਪੂਰਨ kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_ਮੰਜ਼ਿਲ_nibbles(tile_quants, tile_src);
            ਅਚਲ ggml_half * scale_src = (ਅਚਲ ggml_half *) (tile_src + 512);
            ਲਈ (ਪੂਰਨ ਪੰਕਤੀ = 0; ਪੰਕਤੀ < 32; ਪੰਕਤੀ++) {
                ਅਚਲ int64_t r = ct * 32 + ਪੰਕਤੀ;
                ਜੇ (r >= ਸ਼ੁਰੂ_ਪੰਕਤੀ && r < ਸਿਰਾ_ਪੰਕਤੀ && kt < ne0 / 32) {
                    ਖੰਡ_q4_0 & b = dst_ਟੁਕੜਾ[(r - ਸ਼ੁਰੂ_ਪੰਕਤੀ) * ਪੰਕਤੀ_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[ਪੰਕਤੀ]);
                    b.d = scale_src[ਪੰਕਤੀ];
                }
            }
        });
}

ਸਥਿਰ ਖਾਲੀ repack_q4_1_tiled(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    repack_ਮੰਜ਼ਿਲ_tiled_grid_32<ਖੰਡ_q4_1, 1>(t, ਅੰਕੜੇ, ਵਿੱਥ, size, HTP_TENSOR_TILE_ਮਾਪ_Q4_1,
        [](uint8_t * tile_dst, ਅਚਲ ਖੰਡ_q4_1 * src_ਟੁਕੜਾ, int64_t ne0, int64_t ne1, int64_t ਪੰਕਤੀ_stride, ਪੂਰਨ ct, ਪੂਰਨ kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
#ਜੇ defined(__ARM_NEON) || defined(__aarch64__)
            ਜੇ (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                ਲਈ (ਪੂਰਨ row0 = 0; row0 < 32; row0 += 8) {
                    ਅਚਲ uint8_t * qs[8];
                    ਲਈ (ਪੂਰਨ rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_ਟੁਕੜਾ[(ct * 32 + row0 + rr) * ਪੰਕਤੀ_stride + kt].qs;
                    }
                    repack_neon_4bit_ਪੰਕਤੀਆਂ(qs, tile_dst, row0);
                }

                ਲਈ (ਪੂਰਨ ਪੰਕਤੀ = 0; ਪੰਕਤੀ < 32; ++ਪੰਕਤੀ) {
                    ਅਚਲ ਖੰਡ_q4_1 & src_ਖੰਡ = src_ਟੁਕੜਾ[(ct * 32 + ਪੰਕਤੀ) * ਪੰਕਤੀ_stride + kt];
                    scale_dst[2 * ਪੰਕਤੀ + 0] = src_ਖੰਡ.d;
                    scale_dst[2 * ਪੰਕਤੀ + 1] = src_ਖੰਡ.m;
                }
                ਵਾਪਸ;
            }
#ਜੇ_ਅੰਤ
            uint8_t tile_quants[32][32];
            ਲਈ (ਪੂਰਨ ਪੰਕਤੀ = 0; ਪੰਕਤੀ < 32; ਪੰਕਤੀ++) {
                ਅਚਲ int64_t r = ct * 32 + ਪੰਕਤੀ;
                ਜੇ (r < ne1 && kt < ne0 / 32) {
                    ਅਚਲ ਖੰਡ_q4_1 & b = src_ਟੁਕੜਾ[r * ਪੰਕਤੀ_stride + kt];
                    unpack_nibbles_32(tile_quants[ਪੰਕਤੀ], b.qs);
                    scale_dst[2 * ਪੰਕਤੀ + 0] = b.d;
                    scale_dst[2 * ਪੰਕਤੀ + 1] = b.m;
                } ਨਹੀਂਤਾਂ {
                    memset(tile_quants[ਪੰਕਤੀ], 0, 32);
                    scale_dst[2 * ਪੰਕਤੀ + 0] = 0;
                    scale_dst[2 * ਪੰਕਤੀ + 1] = 0;
                }
            }
            transpose_nibbles_ਮੰਜ਼ਿਲ_tile(tile_dst, tile_quants);
        });
}

ਸਥਿਰ ਖਾਲੀ repack_tiled_q4_1(ਖਾਲੀ * ਅੰਕੜੇ, ਅਚਲ ggml_tensor * t, size_t ਵਿੱਥ, size_t size) {
    unpack_from_tiled_grid_32<ਖੰਡ_q4_1, 1>(ਅੰਕੜੇ, t, ਵਿੱਥ, size, HTP_TENSOR_TILE_ਮਾਪ_Q4_1,
        [](ਖੰਡ_q4_1 * dst_ਟੁਕੜਾ, ਅਚਲ uint8_t * tile_src, int64_t ne0, int64_t ਸ਼ੁਰੂ_ਪੰਕਤੀ, int64_t ਸਿਰਾ_ਪੰਕਤੀ, int64_t ਪੰਕਤੀ_stride, ਪੂਰਨ ct, ਪੂਰਨ kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_ਮੰਜ਼ਿਲ_nibbles(tile_quants, tile_src);
            ਅਚਲ ggml_half * scale_src = (ਅਚਲ ggml_half *) (tile_src + 512);
            ਲਈ (ਪੂਰਨ ਪੰਕਤੀ = 0; ਪੰਕਤੀ < 32; ਪੰਕਤੀ++) {
                ਅਚਲ int64_t r = ct * 32 + ਪੰਕਤੀ;
                ਜੇ (r >= ਸ਼ੁਰੂ_ਪੰਕਤੀ && r < ਸਿਰਾ_ਪੰਕਤੀ && kt < ne0 / 32) {
                    ਖੰਡ_q4_1 & b = dst_ਟੁਕੜਾ[(r - ਸ਼ੁਰੂ_ਪੰਕਤੀ) * ਪੰਕਤੀ_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[ਪੰਕਤੀ]);
                    b.d = scale_src[2 * ਪੰਕਤੀ + 0];
                    b.m = scale_src[2 * ਪੰਕਤੀ + 1];
                }
            }
        });
}

ਸਥਿਰ ਖਾਲੀ repack_mxfp4_tiled(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    repack_ਮੰਜ਼ਿਲ_tiled_grid_32<ਖੰਡ_mxfp4, 1>(t, ਅੰਕੜੇ, ਵਿੱਥ, size, HTP_TENSOR_TILE_ਮਾਪ_MXFP4,
        [](uint8_t * tile_dst, ਅਚਲ ਖੰਡ_mxfp4 * src_ਟੁਕੜਾ, int64_t ne0, int64_t ne1, int64_t ਪੰਕਤੀ_stride, ਪੂਰਨ ct, ਪੂਰਨ kt) {
            uint8_t * scale_dst = tile_dst + 512;
#ਜੇ defined(__ARM_NEON) || defined(__aarch64__)
            ਜੇ (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                ਲਈ (ਪੂਰਨ row0 = 0; row0 < 32; row0 += 8) {
                    ਅਚਲ uint8_t * qs[8];
                    ਲਈ (ਪੂਰਨ rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_ਟੁਕੜਾ[(ct * 32 + row0 + rr) * ਪੰਕਤੀ_stride + kt].qs;
                    }
                    repack_neon_4bit_ਪੰਕਤੀਆਂ(qs, tile_dst, row0);
                }

                ਲਈ (ਪੂਰਨ ਪੰਕਤੀ = 0; ਪੰਕਤੀ < 32; ++ਪੰਕਤੀ) {
                    scale_dst[ਪੰਕਤੀ] = src_ਟੁਕੜਾ[(ct * 32 + ਪੰਕਤੀ) * ਪੰਕਤੀ_stride + kt].e;
                }
                ਵਾਪਸ;
            }
#ਜੇ_ਅੰਤ
            uint8_t tile_quants[32][32];
            ਲਈ (ਪੂਰਨ ਪੰਕਤੀ = 0; ਪੰਕਤੀ < 32; ਪੰਕਤੀ++) {
                ਅਚਲ int64_t r = ct * 32 + ਪੰਕਤੀ;
                ਜੇ (r < ne1 && kt < ne0 / 32) {
                    ਅਚਲ ਖੰਡ_mxfp4 & b = src_ਟੁਕੜਾ[r * ਪੰਕਤੀ_stride + kt];
                    unpack_nibbles_32(tile_quants[ਪੰਕਤੀ], b.qs);
                    scale_dst[ਪੰਕਤੀ] = b.e;
                } ਨਹੀਂਤਾਂ {
                    memset(tile_quants[ਪੰਕਤੀ], 0, 32);
                    scale_dst[ਪੰਕਤੀ] = 0;
                }
            }
            transpose_nibbles_ਮੰਜ਼ਿਲ_tile(tile_dst, tile_quants);
        });
}

ਸਥਿਰ ਖਾਲੀ repack_tiled_mxfp4(ਖਾਲੀ * ਅੰਕੜੇ, ਅਚਲ ggml_tensor * t, size_t ਵਿੱਥ, size_t size) {
    unpack_from_tiled_grid_32<ਖੰਡ_mxfp4, 1>(ਅੰਕੜੇ, t, ਵਿੱਥ, size, HTP_TENSOR_TILE_ਮਾਪ_MXFP4,
        [](ਖੰਡ_mxfp4 * dst_ਟੁਕੜਾ, ਅਚਲ uint8_t * tile_src, int64_t ne0, int64_t ਸ਼ੁਰੂ_ਪੰਕਤੀ, int64_t ਸਿਰਾ_ਪੰਕਤੀ, int64_t ਪੰਕਤੀ_stride, ਪੂਰਨ ct, ਪੂਰਨ kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_ਮੰਜ਼ਿਲ_nibbles(tile_quants, tile_src);
            ਅਚਲ uint8_t * scale_src = tile_src + 512;
            ਲਈ (ਪੂਰਨ ਪੰਕਤੀ = 0; ਪੰਕਤੀ < 32; ਪੰਕਤੀ++) {
                ਅਚਲ int64_t r = ct * 32 + ਪੰਕਤੀ;
                ਜੇ (r >= ਸ਼ੁਰੂ_ਪੰਕਤੀ && r < ਸਿਰਾ_ਪੰਕਤੀ && kt < ne0 / 32) {
                    ਖੰਡ_mxfp4 & b = dst_ਟੁਕੜਾ[(r - ਸ਼ੁਰੂ_ਪੰਕਤੀ) * ਪੰਕਤੀ_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[ਪੰਕਤੀ]);
                    b.e = scale_src[ਪੰਕਤੀ];
                }
            }
        });
}

ਸਥਿਰ ਖਾਲੀ repack_q8_0_tiled(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    repack_ਮੰਜ਼ਿਲ_tiled_grid_32<ਖੰਡ_q8_0, 1>(t, ਅੰਕੜੇ, ਵਿੱਥ, size, HTP_TENSOR_TILE_ਮਾਪ_Q8_0,
        [](uint8_t * tile_dst, ਅਚਲ ਖੰਡ_q8_0 * src_ਟੁਕੜਾ, int64_t ne0, int64_t ne1, int64_t ਪੰਕਤੀ_stride, ਪੂਰਨ ct, ਪੂਰਨ kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 1024);
#ਜੇ defined(__ARM_NEON) || defined(__aarch64__)
            ਜੇ (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                ਲਈ (ਪੂਰਨ row0 = 0; row0 < 32; row0 += 8) {
                    ਅਚਲ uint8_t * qs[8];
                    ਲਈ (ਪੂਰਨ rr = 0; rr < 8; ++rr) {
                        qs[rr] = (ਅਚਲ uint8_t *) src_ਟੁਕੜਾ[(ct * 32 + row0 + rr) * ਪੰਕਤੀ_stride + kt].qs;
                    }
                    repack_neon_q8_ਪੰਕਤੀਆਂ(qs, tile_dst, row0);
                }

                ਲਈ (ਪੂਰਨ ਪੰਕਤੀ = 0; ਪੰਕਤੀ < 32; ++ਪੰਕਤੀ) {
                    scale_dst[ਪੰਕਤੀ] = src_ਟੁਕੜਾ[(ct * 32 + ਪੰਕਤੀ) * ਪੰਕਤੀ_stride + kt].d;
                }
                ਵਾਪਸ;
            }
#ਜੇ_ਅੰਤ
            ਲਈ (ਪੂਰਨ cp = 0; cp < 16; cp++) {
                ਅਚਲ ਪੂਰਨ col0 = cp * 2;
                ਅਚਲ ਪੂਰਨ col1 = col0 + 1;
                ਲਈ (ਪੂਰਨ ਪੰਕਤੀ = 0; ਪੰਕਤੀ < 32; ਪੰਕਤੀ++) {
                    ਅਚਲ int64_t r = ct * 32 + ਪੰਕਤੀ;
                    ਅਚਲ ਖੰਡ_q8_0 * b = (r < ne1 && kt < ne0 / 32) ? &src_ਟੁਕੜਾ[r * ਪੰਕਤੀ_stride + kt] : ਖਾਲੀ_ਸੂਚਕ;
                    tile_dst[cp * 64 + 2 * ਪੰਕਤੀ + 0] = b ? b->qs[col0] : 0;
                    tile_dst[cp * 64 + 2 * ਪੰਕਤੀ + 1] = b ? b->qs[col1] : 0;
                }
            }
            ਲਈ (ਪੂਰਨ ਪੰਕਤੀ = 0; ਪੰਕਤੀ < 32; ਪੰਕਤੀ++) {
                ਅਚਲ int64_t r = ct * 32 + ਪੰਕਤੀ;
                scale_dst[ਪੰਕਤੀ] = (r < ne1 && kt < ne0 / 32) ? src_ਟੁਕੜਾ[r * ਪੰਕਤੀ_stride + kt].d : 0;
            }
        });
}

ਸਥਿਰ ਖਾਲੀ repack_tiled_q8_0(ਖਾਲੀ * ਅੰਕੜੇ, ਅਚਲ ggml_tensor * t, size_t ਵਿੱਥ, size_t size) {
    unpack_from_tiled_grid_32<ਖੰਡ_q8_0, 1>(ਅੰਕੜੇ, t, ਵਿੱਥ, size, HTP_TENSOR_TILE_ਮਾਪ_Q8_0,
        [](ਖੰਡ_q8_0 * dst_ਟੁਕੜਾ, ਅਚਲ uint8_t * tile_src, int64_t ne0, int64_t ਸ਼ੁਰੂ_ਪੰਕਤੀ, int64_t ਸਿਰਾ_ਪੰਕਤੀ, int64_t ਪੰਕਤੀ_stride, ਪੂਰਨ ct, ਪੂਰਨ kt) {
            ਲਈ (ਪੂਰਨ cp = 0; cp < 16; cp++) {
                ਅਚਲ ਪੂਰਨ col0 = cp * 2;
                ਅਚਲ ਪੂਰਨ col1 = col0 + 1;
                ਲਈ (ਪੂਰਨ ਪੰਕਤੀ = 0; ਪੰਕਤੀ < 32; ਪੰਕਤੀ++) {
                    ਅਚਲ int64_t r = ct * 32 + ਪੰਕਤੀ;
                    ਜੇ (r >= ਸ਼ੁਰੂ_ਪੰਕਤੀ && r < ਸਿਰਾ_ਪੰਕਤੀ && kt < ne0 / 32) {
                        ਖੰਡ_q8_0 & b = dst_ਟੁਕੜਾ[(r - ਸ਼ੁਰੂ_ਪੰਕਤੀ) * ਪੰਕਤੀ_stride + kt];
                        b.qs[col0] = tile_src[cp * 64 + 2 * ਪੰਕਤੀ + 0];
                        b.qs[col1] = tile_src[cp * 64 + 2 * ਪੰਕਤੀ + 1];
                    }
                }
            }
            ਅਚਲ ggml_half * scale_src = (ਅਚਲ ggml_half *) (tile_src + 1024);
            ਲਈ (ਪੂਰਨ ਪੰਕਤੀ = 0; ਪੰਕਤੀ < 32; ਪੰਕਤੀ++) {
                ਅਚਲ int64_t r = ct * 32 + ਪੰਕਤੀ;
                ਜੇ (r >= ਸ਼ੁਰੂ_ਪੰਕਤੀ && r < ਸਿਰਾ_ਪੰਕਤੀ && kt < ne0 / 32) {
                    dst_ਟੁਕੜਾ[(r - ਸ਼ੁਰੂ_ਪੰਕਤੀ) * ਪੰਕਤੀ_stride + kt].d = scale_src[ਪੰਕਤੀ];
                }
            }
        });
}

// --- Float Types (F16, F32, BF16 -> f16_tiled HMX layout) ---

ਸਥਿਰ ਖਾਲੀ repack_q2_0_tiled(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    GGML_ASSERT(ਵਿੱਥ == 0);
    GGML_ASSERT(t->ne[0] % 64 == 0);
    GGML_ASSERT(size == ggml_nbytes(t));

    ਅਚਲ ਖੰਡ_q2_0 * src = (ਅਚਲ ਖੰਡ_q2_0 *) ਅੰਕੜੇ;
    ਅਚਲ int64_t ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = t->ne[0] / 64;
    ਅਚਲ int64_t n_ਖਾਨਾ_tiles = hex_ਗੇੜਾ_ਉੱਪਰ(t->ne[1], 32) / 32;
    ਅਚਲ int64_t n_ਭਾਰ_tiles = ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
    ਅਚਲ size_t ਮੈਟ੍ਰਿਕਸ_ਮਾਪ = (size_t) n_ਖਾਨਾ_tiles * n_ਭਾਰ_tiles * HTP_TENSOR_TILE_ਮਾਪ_Q2_0;

    ਲਈ (int64_t ਟੁਕੜਾ_2 = 0; ਟੁਕੜਾ_2 < t->ne[2] * t->ne[3]; ++ਟੁਕੜਾ_2) {
        ਅਚਲ ਖੰਡ_q2_0 * src_ਟੁਕੜਾ = src + ਟੁਕੜਾ_2 * t->ne[1] * ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
        uint8_t * dst_ਟੁਕੜਾ = (uint8_t *) t->ਅੰਕੜੇ + ਟੁਕੜਾ_2 * ਮੈਟ੍ਰਿਕਸ_ਮਾਪ;
        ਅਚਲ size_t ct_ਬਾਈਟਾਂ = (size_t) n_ਭਾਰ_tiles * HTP_TENSOR_TILE_ਮਾਪ_Q2_0;

        repack_parallel_for_ct(ਮੈਟ੍ਰਿਕਸ_ਮਾਪ, (ਪੂਰਨ) n_ਖਾਨਾ_tiles, [&](ਪੂਰਨ ct0, ਪੂਰਨ ct1) {
            memset(dst_ਟੁਕੜਾ + (size_t) ct0 * ct_ਬਾਈਟਾਂ, 0, (size_t) (ct1 - ct0) * ct_ਬਾਈਟਾਂ);
            ਅਚਲ int64_t r0 = (int64_t) ct0 * 32;
            ਅਚਲ int64_t r1 = (std::min)((int64_t) ct1 * 32, t->ne[1]);
            ਲਈ (int64_t ਪੰਕਤੀ_ਸੂਚਕਾਂਕ = r0; ਪੰਕਤੀ_ਸੂਚਕਾਂਕ < r1; ++ਪੰਕਤੀ_ਸੂਚਕਾਂਕ) {
                ਅਚਲ int64_t ct = ਪੰਕਤੀ_ਸੂਚਕਾਂਕ / 32;
                ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = (ਪੂਰਨ) (ਪੰਕਤੀ_ਸੂਚਕਾਂਕ % 32);
                ਲਈ (int64_t kt = 0; kt < n_ਭਾਰ_tiles; ++kt) {
                    ਅਚਲ ਖੰਡ_q2_0 * ਖੰਡ = src_ਟੁਕੜਾ + ਪੰਕਤੀ_ਸੂਚਕਾਂਕ * ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + kt;
                    uint8_t * tile = dst_ਟੁਕੜਾ + (ct * n_ਭਾਰ_tiles + kt) * HTP_TENSOR_TILE_ਮਾਪ_Q2_0;
#ਜੇ defined(__ARM_NEON) || defined(__aarch64__)
                    repack_neon_q2_0_ਖੰਡ(ਖੰਡ->qs, tile, ਪੰਕਤੀ);
#ਨਹੀਂਤਾਂ
                    ਲਈ (ਪੂਰਨ ਸਮੂਹ = 0; ਸਮੂਹ < 16; ++ਸਮੂਹ) {
                        ਲਈ (ਪੂਰਨ byte = 0; byte < 4; ++byte) {
                            ਅਚਲ ਪੂਰਨ k = 4 * ਸਮੂਹ + byte;
                            ਅਚਲ uint8_t code = (ਖੰਡ->qs[k / 4] >> (2 * (k % 4))) & 3;
                            tile[(ਸਮੂਹ >> 2) * 128 + 4 * ਪੰਕਤੀ + byte] |= code << (2 * (ਸਮੂਹ & 3));
                        }
                    }
#ਜੇ_ਅੰਤ
                    ਸਮ੍ਰਿਤੀ_ਨਕਲ(tile + 512 + 2 * ਪੰਕਤੀ, &ਖੰਡ->d, ਆਕਾਰ(ਖੰਡ->d));
                }
            }
        });
    }
}

ਸਥਿਰ ਖਾਲੀ repack_tiled_q2_0(ਖਾਲੀ * ਅੰਕੜੇ, ਅਚਲ ggml_tensor * t, size_t ਵਿੱਥ, size_t size) {
    GGML_ASSERT(t->ne[0] % 64 == 0);
    GGML_ASSERT(ਵਿੱਥ + size <= ggml_nbytes(t));

    ਅਚਲ size_t ਪਹਿਲਾ_ਖੰਡ = ਵਿੱਥ / ਆਕਾਰ(ਖੰਡ_q2_0);
    ਅਚਲ size_t ਸਿਰਾ_ਖੰਡ = (ਵਿੱਥ + size + ਆਕਾਰ(ਖੰਡ_q2_0) - 1) / ਆਕਾਰ(ਖੰਡ_q2_0);
    ਅਚਲ size_t ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = t->ne[0] / 64;
    ਅਚਲ size_t ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਟੁਕੜਾ = t->ne[1] * ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
    ਅਚਲ size_t n_ਭਾਰ_tiles = ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
    ਅਚਲ size_t n_ਖਾਨਾ_tiles = hex_ਗੇੜਾ_ਉੱਪਰ(t->ne[1], 32) / 32;
    ਅਚਲ size_t ਮੈਟ੍ਰਿਕਸ_ਮਾਪ = n_ਖਾਨਾ_tiles * n_ਭਾਰ_tiles * HTP_TENSOR_TILE_ਮਾਪ_Q2_0;

    ਲਈ (size_t ਖੰਡ_ਸੂਚਕਾਂਕ = ਪਹਿਲਾ_ਖੰਡ; ਖੰਡ_ਸੂਚਕਾਂਕ < ਸਿਰਾ_ਖੰਡ; ++ਖੰਡ_ਸੂਚਕਾਂਕ) {
        ਅਚਲ size_t ਟੁਕੜਾ_2 = ਖੰਡ_ਸੂਚਕਾਂਕ / ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਟੁਕੜਾ;
        ਅਚਲ size_t ਵਿੱਚ_ਟੁਕੜਾ = ਖੰਡ_ਸੂਚਕਾਂਕ % ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਟੁਕੜਾ;
        ਅਚਲ size_t ਪੰਕਤੀ_ਸੂਚਕਾਂਕ = ਵਿੱਚ_ਟੁਕੜਾ / ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
        ਅਚਲ size_t kt = ਵਿੱਚ_ਟੁਕੜਾ % ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
        ਅਚਲ size_t ct = ਪੰਕਤੀ_ਸੂਚਕਾਂਕ / 32;
        ਅਚਲ size_t ਪੰਕਤੀ = ਪੰਕਤੀ_ਸੂਚਕਾਂਕ % 32;
        ਅਚਲ uint8_t * tile = (ਅਚਲ uint8_t *) t->ਅੰਕੜੇ + ਟੁਕੜਾ_2 * ਮੈਟ੍ਰਿਕਸ_ਮਾਪ +
            (ct * n_ਭਾਰ_tiles + kt) * HTP_TENSOR_TILE_ਮਾਪ_Q2_0;

        ਖੰਡ_q2_0 ਖੰਡ = {};
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਖੰਡ.d, tile + 512 + 2 * ਪੰਕਤੀ, ਆਕਾਰ(ਖੰਡ.d));
        ਲਈ (ਪੂਰਨ ਸਮੂਹ = 0; ਸਮੂਹ < 16; ++ਸਮੂਹ) {
            ਲਈ (ਪੂਰਨ byte = 0; byte < 4; ++byte) {
                ਅਚਲ uint8_t packed = tile[(ਸਮੂਹ >> 2) * 128 + 4 * ਪੰਕਤੀ + byte];
                ਅਚਲ uint8_t code = (packed >> (2 * (ਸਮੂਹ & 3))) & 3;
                ਖੰਡ.qs[ਸਮੂਹ] |= code << (2 * byte);
            }
        }

        ਅਚਲ size_t ਖੰਡ_begin = ਖੰਡ_ਸੂਚਕਾਂਕ * ਆਕਾਰ(ਖੰਡ_q2_0);
        ਅਚਲ size_t copy_begin = (std::max)(ਖੰਡ_begin, ਵਿੱਥ);
        ਅਚਲ size_t copy_ਸਿਰਾ = (std::min)(ਖੰਡ_begin + ਆਕਾਰ(ਖੰਡ_q2_0), ਵਿੱਥ + size);
        ਸਮ੍ਰਿਤੀ_ਨਕਲ((uint8_t *) ਅੰਕੜੇ + copy_begin - ਵਿੱਥ, (ਅਚਲ uint8_t *) &ਖੰਡ + copy_begin - ਖੰਡ_begin, copy_ਸਿਰਾ - copy_begin);
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ SrcT, ਕਿਸਮ_ਨਾਂ ਮੰਜ਼ਿਲ_Fp16_Fn>
ਸਥਿਰ ਖਾਲੀ repack_float_ਮੰਜ਼ਿਲ_f16_tiled(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size, ਮੰਜ਼ਿਲ_Fp16_Fn && ਮੰਜ਼ਿਲ_fp16) {
    repack_ਮੰਜ਼ਿਲ_tiled_grid_32<SrcT, 32>(t, ਅੰਕੜੇ, ਵਿੱਥ, size, HTP_TENSOR_TILE_ਮਾਪ_F16,
        [&](uint8_t * tile_ਬਾਈਟਾਂ, ਅਚਲ SrcT * src_ਟੁਕੜਾ, int64_t ne0, int64_t ne1, int64_t ਪੰਕਤੀ_stride, ਪੂਰਨ ct, ਪੂਰਨ kt) {
            uint16_t * tile_dst = (uint16_t *) tile_ਬਾਈਟਾਂ;
#ਜੇ defined(__ARM_NEON) || defined(__aarch64__)
            ਜੇ (ct * 32 + 32 <= ne1 && kt * 32 + 32 <= ne0) {
                repack_neon_f16_tile(tile_dst, src_ਟੁਕੜਾ, ਪੰਕਤੀ_stride, ct, kt);
                ਵਾਪਸ;
            }
#ਜੇ_ਅੰਤ
            ਲਈ (ਪੂਰਨ cp = 0; cp < 16; cp++) {
                ਅਚਲ int64_t c0 = kt * 32 + cp * 2;
                ਅਚਲ int64_t c1 = c0 + 1;
                ਲਈ (ਪੂਰਨ ਪੰਕਤੀ = 0; ਪੰਕਤੀ < 32; ਪੰਕਤੀ++) {
                    ਅਚਲ int64_t r = ct * 32 + ਪੰਕਤੀ;
                    ਅਚਲ SrcT * r_ptr = (r < ne1) ? &src_ਟੁਕੜਾ[r * ਪੰਕਤੀ_stride] : ਖਾਲੀ_ਸੂਚਕ;
                    tile_dst[cp * 64 + ਪੰਕਤੀ * 2 + 0] = (r_ptr && c0 < ne0) ? ਮੰਜ਼ਿਲ_fp16(r_ptr[c0]) : 0;
                    tile_dst[cp * 64 + ਪੰਕਤੀ * 2 + 1] = (r_ptr && c1 < ne0) ? ਮੰਜ਼ਿਲ_fp16(r_ptr[c1]) : 0;
                }
            }
        });
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ DstT, ਕਿਸਮ_ਨਾਂ FromFp16Fn>
ਸਥਿਰ ਖਾਲੀ unpack_f16_tiled_ਮੰਜ਼ਿਲ_float(ਖਾਲੀ * ਅੰਕੜੇ, ਅਚਲ ggml_tensor * t, size_t ਵਿੱਥ, size_t size, FromFp16Fn && from_fp16) {
    unpack_from_tiled_grid_32<DstT, 32>(ਅੰਕੜੇ, t, ਵਿੱਥ, size, HTP_TENSOR_TILE_ਮਾਪ_F16,
        [&](DstT * dst_ਟੁਕੜਾ, ਅਚਲ uint8_t * tile_ਬਾਈਟਾਂ, int64_t ne0, int64_t ਸ਼ੁਰੂ_ਪੰਕਤੀ, int64_t ਸਿਰਾ_ਪੰਕਤੀ, int64_t ਪੰਕਤੀ_stride, ਪੂਰਨ ct, ਪੂਰਨ kt) {
            ਅਚਲ uint16_t * tile_src = (ਅਚਲ uint16_t *) tile_ਬਾਈਟਾਂ;
            ਲਈ (ਪੂਰਨ cp = 0; cp < 16; cp++) {
                ਅਚਲ int64_t c0 = kt * 32 + cp * 2;
                ਅਚਲ int64_t c1 = c0 + 1;
                ਲਈ (ਪੂਰਨ ਪੰਕਤੀ = 0; ਪੰਕਤੀ < 32; ਪੰਕਤੀ++) {
                    ਅਚਲ int64_t r = ct * 32 + ਪੰਕਤੀ;
                    ਜੇ (r >= ਸ਼ੁਰੂ_ਪੰਕਤੀ && r < ਸਿਰਾ_ਪੰਕਤੀ) {
                        DstT * r_ptr = dst_ਟੁਕੜਾ + (r - ਸ਼ੁਰੂ_ਪੰਕਤੀ) * ਪੰਕਤੀ_stride;
                        ਜੇ (c0 < ne0) r_ptr[c0] = from_fp16(tile_src[cp * 64 + ਪੰਕਤੀ * 2 + 0]);
                        ਜੇ (c1 < ne0) r_ptr[c1] = from_fp16(tile_src[cp * 64 + ਪੰਕਤੀ * 2 + 1]);
                    }
                }
            }
        });
}

// --- Generic K-Quant (super-block 256) grid walkers ---

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ ਖੰਡ_T, ਕਿਸਮ_ਨਾਂ PackKtFn>
ਸਥਿਰ ਖਾਲੀ repack_kquant_ਮੰਜ਼ਿਲ_tiled_grid(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size,
                                        size_t tile_ਮਾਪ, PackKtFn && pack_kt) {
    GGML_ASSERT(ਵਿੱਥ == 0);
    GGML_ਅਣਵਰਤਿਆ(size);

    ਅਚਲ ਖੰਡ_T * src_ਮੈਟ੍ਰਿਕਸ = (ਅਚਲ ਖੰਡ_T *) ਅੰਕੜੇ;
    ਅਚਲ int64_t ne0 = t->ne[0];
    ਅਚਲ int64_t ne1 = t->ne[1];
    ਅਚਲ int64_t ne2 = t->ne[2];
    ਅਚਲ int64_t ne3 = t->ne[3];
    GGML_ASSERT(ne0 % QK_K == 0);

    ਅਚਲ ਪੂਰਨ n_ਖਾਨਾ_tiles    = hex_ਗੇੜਾ_ਉੱਪਰ(ne1, 32) / 32;
    ਅਚਲ ਪੂਰਨ n_k_tiles      = hex_ਗੇੜਾ_ਉੱਪਰ(ne0, 32) / 32;
    ਅਚਲ size_t ਮੈਟ੍ਰਿਕਸ_ਮਾਪ = (size_t) n_ਖਾਨਾ_tiles * n_k_tiles * tile_ਮਾਪ;
    ਅਚਲ int64_t sb_ਪ੍ਰਤੀ_ਪੰਕਤੀ = ne0 / QK_K;

    ਲਈ (int64_t ਟੁਕੜਾ_idx = 0; ਟੁਕੜਾ_idx < ne2 * ne3; ਟੁਕੜਾ_idx++) {
        ਅਚਲ ਖੰਡ_T * src_ਟੁਕੜਾ = src_ਮੈਟ੍ਰਿਕਸ + ਟੁਕੜਾ_idx * (ne1 * sb_ਪ੍ਰਤੀ_ਪੰਕਤੀ);
        uint8_t * ਮੈਟ੍ਰਿਕਸ_dst     = (uint8_t *) t->ਅੰਕੜੇ + ਟੁਕੜਾ_idx * ਮੈਟ੍ਰਿਕਸ_ਮਾਪ;
        ਅਚਲ size_t ct_ਬਾਈਟਾਂ    = (size_t) n_k_tiles * tile_ਮਾਪ;

        repack_parallel_for_ct(ਮੈਟ੍ਰਿਕਸ_ਮਾਪ, n_ਖਾਨਾ_tiles, [&](ਪੂਰਨ ct0, ਪੂਰਨ ct1) {
            // Padding rows and OR-ed bit planes require zeroed tiles
            memset(ਮੈਟ੍ਰਿਕਸ_dst + (size_t) ct0 * ct_ਬਾਈਟਾਂ, 0, (size_t) (ct1 - ct0) * ct_ਬਾਈਟਾਂ);

            ਅਚਲ int64_t r0 = (int64_t) ct0 * 32;
            ਅਚਲ int64_t r1 = (std::min)((int64_t) ct1 * 32, ne1);
            ਲਈ (int64_t r = r0; r < r1; r++) {
                ਅਚਲ ਪੂਰਨ ct  = (ਪੂਰਨ) (r / 32);
                ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = (ਪੂਰਨ) (r % 32);
                ਅਚਲ ਖੰਡ_T * src_ਪੰਕਤੀ = src_ਟੁਕੜਾ + r * sb_ਪ੍ਰਤੀ_ਪੰਕਤੀ;

                ਲਈ (ਪੂਰਨ kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile = ਮੈਟ੍ਰਿਕਸ_dst + ((size_t) ct * n_k_tiles + kt) * tile_ਮਾਪ;
                    pack_kt(tile, &src_ਪੰਕਤੀ[kt / 8], ਪੰਕਤੀ, kt % 8);
                }
            }
        });
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ ਖੰਡ_T, ਕਿਸਮ_ਨਾਂ UnpackSbFn>
ਸਥਿਰ ਖਾਲੀ unpack_kquant_from_tiled_grid(ਖਾਲੀ * ਅੰਕੜੇ, ਅਚਲ ggml_tensor * t, size_t ਵਿੱਥ, size_t size,
                                          size_t tile_ਮਾਪ, UnpackSbFn && unpack_sb) {
    uint8_t * dst_ਬਾਈਟਾਂ = (uint8_t *) ਅੰਕੜੇ;
    ਅਚਲ int64_t ne0 = t->ne[0];
    ਅਚਲ int64_t ne1 = t->ne[1];
    ਅਚਲ int64_t ne2 = t->ne[2];
    ਅਚਲ int64_t ne3 = t->ne[3];
    GGML_ASSERT(ne0 % QK_K == 0);

    ਅਚਲ ਪੂਰਨ n_ਖਾਨਾ_tiles    = hex_ਗੇੜਾ_ਉੱਪਰ(ne1, 32) / 32;
    ਅਚਲ ਪੂਰਨ n_k_tiles      = hex_ਗੇੜਾ_ਉੱਪਰ(ne0, 32) / 32;
    ਅਚਲ size_t ਮੈਟ੍ਰਿਕਸ_ਮਾਪ = (size_t) n_ਖਾਨਾ_tiles * n_k_tiles * tile_ਮਾਪ;
    ਅਚਲ int64_t sb_ਪ੍ਰਤੀ_ਪੰਕਤੀ = ne0 / QK_K;

    ਅਚਲ size_t ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ = ggml_ਪੰਕਤੀ_ਮਾਪ(t->type, ne0);
    ਅਚਲ size_t ਟੁਕੜਾ_ਮਾਪ     = (size_t) ne1 * ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ;
    ਅਚਲ int64_t ਸ਼ੁਰੂ_ਟੁਕੜਾ   = ਵਿੱਥ / ਟੁਕੜਾ_ਮਾਪ;
    int64_t ਸਿਰਾ_ਟੁਕੜਾ           = (ਵਿੱਥ + size + ਟੁਕੜਾ_ਮਾਪ - 1) / ਟੁਕੜਾ_ਮਾਪ;
    ਜੇ (ਸਿਰਾ_ਟੁਕੜਾ > ne2 * ne3) {
        ਸਿਰਾ_ਟੁਕੜਾ = ne2 * ne3;
    }

    ਲਈ (int64_t ਟੁਕੜਾ_idx = ਸ਼ੁਰੂ_ਟੁਕੜਾ; ਟੁਕੜਾ_idx < ਸਿਰਾ_ਟੁਕੜਾ; ਟੁਕੜਾ_idx++) {
        ਅਚਲ size_t cur_ਸ਼ੁਰੂ_byte     = (std::max)(ਵਿੱਥ, (size_t) ਟੁਕੜਾ_idx * ਟੁਕੜਾ_ਮਾਪ);
        ਅਚਲ size_t cur_ਸਿਰਾ_byte       = (std::min)(ਵਿੱਥ + size, (size_t) (ਟੁਕੜਾ_idx + 1) * ਟੁਕੜਾ_ਮਾਪ);
        ਅਚਲ size_t ਟੁਕੜਾ_ਵਿੱਥ_ਸ਼ੁਰੂ = cur_ਸ਼ੁਰੂ_byte - (size_t) ਟੁਕੜਾ_idx * ਟੁਕੜਾ_ਮਾਪ;
        ਅਚਲ size_t ਟੁਕੜਾ_ਵਿੱਥ_ਸਿਰਾ   = cur_ਸਿਰਾ_byte - (size_t) ਟੁਕੜਾ_idx * ਟੁਕੜਾ_ਮਾਪ;

        ਅਚਲ int64_t ਸ਼ੁਰੂ_ਪੰਕਤੀ = ਟੁਕੜਾ_ਵਿੱਥ_ਸ਼ੁਰੂ / ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ;
        ਅਚਲ int64_t ਸਿਰਾ_ਪੰਕਤੀ   = (std::min)((int64_t) ((ਟੁਕੜਾ_ਵਿੱਥ_ਸਿਰਾ + ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ - 1) / ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ), ne1);

        ਖੰਡ_T * dst_ਟੁਕੜਾ         = (ਖੰਡ_T *) (dst_ਬਾਈਟਾਂ + (cur_ਸ਼ੁਰੂ_byte - ਵਿੱਥ));
        ਅਚਲ uint8_t * ਮੈਟ੍ਰਿਕਸ_src = (ਅਚਲ uint8_t *) t->ਅੰਕੜੇ + ਟੁਕੜਾ_idx * ਮੈਟ੍ਰਿਕਸ_ਮਾਪ;

        ਲਈ (int64_t r = ਸ਼ੁਰੂ_ਪੰਕਤੀ; r < ਸਿਰਾ_ਪੰਕਤੀ; r++) {
            ਅਚਲ ਪੂਰਨ ct  = (ਪੂਰਨ) (r / 32);
            ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = (ਪੂਰਨ) (r % 32);
            ਖੰਡ_T * dst_ਪੰਕਤੀ = dst_ਟੁਕੜਾ + (r - ਸ਼ੁਰੂ_ਪੰਕਤੀ) * sb_ਪ੍ਰਤੀ_ਪੰਕਤੀ;

            ਲਈ (int64_t sb = 0; sb < sb_ਪ੍ਰਤੀ_ਪੰਕਤੀ; sb++) {
                ਖੰਡ_T * b = &dst_ਪੰਕਤੀ[sb];
                memset(b, 0, ਆਕਾਰ(ਖੰਡ_T));
                ਅਚਲ uint8_t * sb_tiles = ਮੈਟ੍ਰਿਕਸ_src + ((size_t) ct * n_k_tiles + sb * 8) * tile_ਮਾਪ;
                unpack_sb(b, sb_tiles, tile_ਮਾਪ, ਪੰਕਤੀ);
            }
        }
    }
}

// --- K-Quant Helpers & Per-Type Implementations (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) ---

#ਜੇ !defined(__ARM_NEON) && !defined(__aarch64__)
// unsigned 6-bit value (0..63) of element e of a Q6_K block, same bit layout as dequantize_row_q6_K
ਸਥਿਰ ਸਤਰਅੰਦਰ uint8_t q6_K_get_quant(ਅਚਲ ਖੰਡ_q6_K * b, ਪੂਰਨ e) {
    ਅਚਲ ਪੂਰਨ c = e / 128;
    ਅਚਲ ਪੂਰਨ w = e % 128;
    ਅਚਲ ਪੂਰਨ g = w / 32;
    ਅਚਲ ਪੂਰਨ l = w % 32;
    ਅਚਲ uint8_t * ql = b->ql + c * 64;
    ਅਚਲ uint8_t * qh = b->qh + c * 32;
    uint8_t lo, hi;
    ਚੋਣ (g) {
        ਹਾਲਤ 0:  lo = ql[l]      & 0xF; hi = (qh[l] >> 0) & 3; ਰੋਕੋ;
        ਹਾਲਤ 1:  lo = ql[l + 32] & 0xF; hi = (qh[l] >> 2) & 3; ਰੋਕੋ;
        ਹਾਲਤ 2:  lo = ql[l]      >> 4;  hi = (qh[l] >> 4) & 3; ਰੋਕੋ;
        ਮੂਲ: lo = ql[l + 32] >> 4;  hi = (qh[l] >> 6) & 3; ਰੋਕੋ;
    }
    ਵਾਪਸ (uint8_t) (lo | (hi << 4));
}
#ਜੇ_ਅੰਤ

ਸਥਿਰ ਖਾਲੀ repack_q6_K_tiled(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    repack_kquant_ਮੰਜ਼ਿਲ_tiled_grid<ਖੰਡ_q6_K>(t, ਅੰਕੜੇ, ਵਿੱਥ, size, HTP_TENSOR_TILE_ਮਾਪ_Q6_K,
        [](uint8_t * tile, ਅਚਲ ਖੰਡ_q6_K * b, ਪੂਰਨ ਪੰਕਤੀ, ਪੂਰਨ kt_local) {
            ਅਚਲ ਦਸ਼ਮਲਵ d     = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(b->d);
            uint8_t * lo_pl   = tile;
            uint8_t * hi_pl   = tile + 512;
            ggml_half * sc_pl = (ggml_half *) (tile + 768);

#ਜੇ defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q6_k(b, kt_local, lo_pl, hi_pl, ਪੰਕਤੀ);
#ਨਹੀਂਤਾਂ
            ਲਈ (ਪੂਰਨ lk = 0; lk < 32; lk++) {
                ਅਚਲ uint8_t q6 = q6_K_get_quant(b, kt_local * 32 + lk);
                ਅਚਲ ਪੂਰਨ g   = lk >> 2;
                ਅਚਲ ਪੂਰਨ pos = ਪੰਕਤੀ * 4 + (lk & 3);
                lo_pl[(g >> 1) * 128 + pos] |= (uint8_t) ((q6 & 0xF) << ((g & 1) * 4));
                hi_pl[(g >> 2) * 128 + pos] |= (uint8_t) ((q6 >> 4) << ((g & 3) * 2));
            }
#ਜੇ_ਅੰਤ
            ਲਈ (ਪੂਰਨ sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + ਪੰਕਤੀ] = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(d * (ਦਸ਼ਮਲਵ) b->scales[kt_local * 2 + sub]);
            }
        });
}

ਸਥਿਰ ਖਾਲੀ repack_tiled_q6_K(ਖਾਲੀ * ਅੰਕੜੇ, ਅਚਲ ggml_tensor * t, size_t ਵਿੱਥ, size_t size) {
    unpack_kquant_from_tiled_grid<ਖੰਡ_q6_K>(ਅੰਕੜੇ, t, ਵਿੱਥ, size, HTP_TENSOR_TILE_ਮਾਪ_Q6_K,
        [](ਖੰਡ_q6_K * b, ਅਚਲ uint8_t * sb_tiles, size_t tile_ਮਾਪ, ਪੂਰਨ ਪੰਕਤੀ) {
            ਦਸ਼ਮਲਵ sub_scales[16];
            ਲਈ (ਪੂਰਨ kt_local = 0; kt_local < 8; kt_local++) {
                ਅਚਲ uint8_t *   tile  = sb_tiles + kt_local * tile_ਮਾਪ;
                ਅਚਲ uint8_t *   lo_pl = tile;
                ਅਚਲ uint8_t *   hi_pl = tile + 512;
                ਅਚਲ ggml_half * sc_pl = (ਅਚਲ ggml_half *) (tile + 768);

                ਅਚਲ ਪੂਰਨ c  = kt_local / 4;
                ਅਚਲ ਪੂਰਨ g  = kt_local % 4;
                uint8_t * ql = b->ql + c * 64;
                uint8_t * qh = b->qh + c * 32;

                ਲਈ (ਪੂਰਨ lk = 0; lk < 32; lk++) {
                    ਅਚਲ ਪੂਰਨ g_tile = lk >> 2;
                    ਅਚਲ ਪੂਰਨ pos    = ਪੰਕਤੀ * 4 + (lk & 3);
                    ਅਚਲ uint8_t lo = (lo_pl[(g_tile >> 1) * 128 + pos] >> ((g_tile & 1) * 4)) & 0xF;
                    ਅਚਲ uint8_t hi = (hi_pl[(g_tile >> 2) * 128 + pos] >> ((g_tile & 3) * 2)) & 3;

                    ਚੋਣ (g) {
                        ਹਾਲਤ 0:  ql[lk]      |= lo;        qh[lk] |= (hi << 0); ਰੋਕੋ;
                        ਹਾਲਤ 1:  ql[lk + 32] |= lo;        qh[lk] |= (hi << 2); ਰੋਕੋ;
                        ਹਾਲਤ 2:  ql[lk]      |= (lo << 4); qh[lk] |= (hi << 4); ਰੋਕੋ;
                        ਮੂਲ: ql[lk + 32] |= (lo << 4); qh[lk] |= (hi << 6); ਰੋਕੋ;
                    }
                }

                ਲਈ (ਪੂਰਨ sub = 0; sub < 2; sub++) {
                    sub_scales[kt_local * 2 + sub] = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(sc_pl[sub * 32 + ਪੰਕਤੀ]);
                }
            }

            ਦਸ਼ਮਲਵ max_abs_scale = 0.0f;
            ਲਈ (ਪੂਰਨ s = 0; s < 16; s++) {
                max_abs_scale = (std::max)(max_abs_scale, fabsf(sub_scales[s]));
            }

            ਜੇ (max_abs_scale == 0.0f) {
                b->d = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(0.0f);
            } ਨਹੀਂਤਾਂ {
                b->d = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(max_abs_scale / 127.0f);
                ਅਚਲ ਦਸ਼ਮਲਵ d_actual = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(b->d);
                ਅਚਲ ਦਸ਼ਮਲਵ inv_d    = (d_actual != 0.0f) ? (1.0f / d_actual) : 0.0f;
                ਲਈ (ਪੂਰਨ s = 0; s < 16; s++) {
                    ਅਚਲ ਪੂਰਨ sc = (ਪੂਰਨ) roundf(sub_scales[s] * inv_d);
                    b->scales[s] = (int8_t) (std::max)(-128, (std::min)(127, sc));
                }
            }
        });
}

#ਜੇ !defined(__ARM_NEON) && !defined(__aarch64__)
// low 2 bits (0..3) of element e of a Q2_K or Q3_K block, same bit layout as dequantize_row_q2_K / q3_K
ਸਥਿਰ ਸਤਰਅੰਦਰ uint8_t q2_3_K_get_low2(ਅਚਲ uint8_t * qs, ਪੂਰਨ e) {
    ਅਚਲ ਪੂਰਨ c = e / 128;
    ਅਚਲ ਪੂਰਨ j = (e % 128) / 32;
    ਅਚਲ ਪੂਰਨ l = e % 32;
    ਵਾਪਸ (qs[c * 32 + l] >> (2 * j)) & 3;
}

// hmask bit of element e of a Q3_K block, same bit layout as dequantize_row_q3_K
ਸਥਿਰ ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ q3_K_get_hbit(ਅਚਲ ਖੰਡ_q3_K * b, ਪੂਰਨ e) {
    ਅਚਲ ਪੂਰਨ c = e / 128;
    ਅਚਲ ਪੂਰਨ j = (e % 128) / 32;
    ਅਚਲ ਪੂਰਨ l = e % 32;
    ਵਾਪਸ (b->hmask[l] >> (c * 4 + j)) & 1;
}
#ਜੇ_ਅੰਤ

// signed 6-bit scale j (-32..31) of a Q3_K block, same packing as quantize_row_q3_K_ref
ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ q3_K_get_scale(ਅਚਲ uint8_t * scales, ਪੂਰਨ j) {
    ਅਚਲ ਪੂਰਨ lo = (j < 8) ? (scales[j] & 0xF) : (scales[j - 8] >> 4);
    ਅਚਲ ਪੂਰਨ hi = (scales[8 + j % 4] >> (2 * (j / 4))) & 3;
    ਵਾਪਸ (lo | (hi << 4)) - 32;
}

// read-back: find fp16 d and l[j] in [lmin, lmax] with fp16(d * l[j]) == prod[j] for all j, false if none
ਸਥਿਰ ਤਾਰਕਿਕ hexagon_recover_k_scales(ਅਚਲ ggml_half * prod, ਪੂਰਨ n, ਪੂਰਨ lmin, ਪੂਰਨ lmax, ggml_half * d_ਬਾਹਰਲਾ, ਪੂਰਨ * l_ਬਾਹਰਲਾ) {
    ਪੂਰਨ jmax = 0;
    ਲਈ (ਪੂਰਨ j = 1; j < n; j++) {
        ਜੇ (fabsf(GGML_FP16_ਮੰਜ਼ਿਲ_FP32(prod[j])) > fabsf(GGML_FP16_ਮੰਜ਼ਿਲ_FP32(prod[jmax]))) {
            jmax = j;
        }
    }
    ਅਚਲ ਦਸ਼ਮਲਵ pmax = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(prod[jmax]);
    ਜੇ (pmax == 0.0f) {
        *d_ਬਾਹਰਲਾ = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(0.0f);
        ਲਈ (ਪੂਰਨ j = 0; j < n; j++) {
            l_ਬਾਹਰਲਾ[j] = 0;
        }
        ਵਾਪਸ ਸੱਚ;
    }

    // the quantizers put the largest scale at or near the range end, so try large |l| first
    ਅਚਲ ਪੂਰਨ lext = (std::max)(-lmin, lmax);
    ਲਈ (ਪੂਰਨ a = lext; a >= 1; a--) {
        ਲਈ (ਪੂਰਨ sign : { 1, -1 }) {
            ਅਚਲ ਪੂਰਨ lj = sign * a;
            ਜੇ (lj < lmin || lj > lmax) {
                ਜਾਰੀ;
            }
            ਅਚਲ ggml_half d0 = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(pmax / (ਦਸ਼ਮਲਵ) lj);
            ਲਈ (ਪੂਰਨ ulp : { 0, -1, 1 }) {
                ggml_half d = d0;
                uint16_t ਬਿੱਟ;
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਬਿੱਟ, &d, ਆਕਾਰ(ਬਿੱਟ));
                ਬਿੱਟ = (uint16_t) (ਬਿੱਟ + ulp);
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(&d, &ਬਿੱਟ, ਆਕਾਰ(ਬਿੱਟ));

                ਅਚਲ ਦਸ਼ਮਲਵ df = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(d);
                ਜੇ (!std::isfinite(df) || df == 0.0f) {
                    ਜਾਰੀ;
                }
                ਤਾਰਕਿਕ ok = ਸੱਚ;
                ਲਈ (ਪੂਰਨ j = 0; j < n && ok; j++) {
                    ਅਚਲ ਪੂਰਨ l = (ਪੂਰਨ) roundf(GGML_FP16_ਮੰਜ਼ਿਲ_FP32(prod[j]) / df);
                    ਅਚਲ ggml_half p = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(df * (ਦਸ਼ਮਲਵ) l);
                    ok = l >= lmin && l <= lmax && memcmp(&p, &prod[j], ਆਕਾਰ(p)) == 0;
                    l_ਬਾਹਰਲਾ[j] = l;
                }
                ਜੇ (ok) {
                    *d_ਬਾਹਰਲਾ = d;
                    ਵਾਪਸ ਸੱਚ;
                }
            }
        }
    }
    ਵਾਪਸ ਝੂਠ;
}

ਸਥਿਰ ਖਾਲੀ repack_q3_K_tiled(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    repack_kquant_ਮੰਜ਼ਿਲ_tiled_grid<ਖੰਡ_q3_K>(t, ਅੰਕੜੇ, ਵਿੱਥ, size, HTP_TENSOR_TILE_ਮਾਪ_Q3_K,
        [](uint8_t * tile, ਅਚਲ ਖੰਡ_q3_K * b, ਪੂਰਨ ਪੰਕਤੀ, ਪੂਰਨ kt_local) {
            ਅਚਲ ਦਸ਼ਮਲਵ d     = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(b->d);
            uint8_t * lo_pl   = tile;
            uint8_t * neg_pl  = tile + 256;
            ggml_half * sc_pl = (ggml_half *) (tile + 384);

#ਜੇ defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q2_3_k_low2(b->qs, kt_local, lo_pl, ਪੰਕਤੀ);
            repack_neon_q3_k_hmask(b->hmask, kt_local, neg_pl, ਪੰਕਤੀ);
#ਨਹੀਂਤਾਂ
            ਲਈ (ਪੂਰਨ lk = 0; lk < 32; lk++) {
                ਅਚਲ ਪੂਰਨ e   = kt_local * 32 + lk;
                ਅਚਲ ਪੂਰਨ g   = lk >> 2;
                ਅਚਲ ਪੂਰਨ pos = ਪੰਕਤੀ * 4 + (lk & 3);
                lo_pl[(g >> 2) * 128 + pos] |= (uint8_t) (q2_3_K_get_low2(b->qs, e) << ((g & 3) * 2));
                ਜੇ (!q3_K_get_hbit(b, e)) {
                    neg_pl[pos] |= (uint8_t) (1 << g);
                }
            }
#ਜੇ_ਅੰਤ
            ਲਈ (ਪੂਰਨ sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + ਪੰਕਤੀ] = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(d * (ਦਸ਼ਮਲਵ) q3_K_get_scale(b->scales, kt_local * 2 + sub));
            }
        });
}

ਸਥਿਰ ਖਾਲੀ repack_tiled_q3_K(ਖਾਲੀ * ਅੰਕੜੇ, ਅਚਲ ggml_tensor * t, size_t ਵਿੱਥ, size_t size) {
    unpack_kquant_from_tiled_grid<ਖੰਡ_q3_K>(ਅੰਕੜੇ, t, ਵਿੱਥ, size, HTP_TENSOR_TILE_ਮਾਪ_Q3_K,
        [](ਖੰਡ_q3_K * b, ਅਚਲ uint8_t * sb_tiles, size_t tile_ਮਾਪ, ਪੂਰਨ ਪੰਕਤੀ) {
            ggml_half sub_scales[16];
            ਲਈ (ਪੂਰਨ kt_local = 0; kt_local < 8; kt_local++) {
                ਅਚਲ uint8_t *   tile   = sb_tiles + kt_local * tile_ਮਾਪ;
                ਅਚਲ uint8_t *   lo_pl  = tile;
                ਅਚਲ uint8_t *   neg_pl = tile + 256;
                ਅਚਲ ggml_half * sc_pl  = (ਅਚਲ ggml_half *) (tile + 384);

                ਲਈ (ਪੂਰਨ lk = 0; lk < 32; lk++) {
                    ਅਚਲ ਪੂਰਨ e   = kt_local * 32 + lk;
                    ਅਚਲ ਪੂਰਨ g   = lk >> 2;
                    ਅਚਲ ਪੂਰਨ pos = ਪੰਕਤੀ * 4 + (lk & 3);
                    ਅਚਲ uint8_t lo = (lo_pl[(g >> 2) * 128 + pos] >> ((g & 3) * 2)) & 3;

                    ਅਚਲ ਪੂਰਨ c = e / 128;
                    ਅਚਲ ਪੂਰਨ j = (e % 128) / 32;
                    ਅਚਲ ਪੂਰਨ l = e % 32;
                    b->qs[c * 32 + l] |= (uint8_t) (lo << (2 * j));
                    ਜੇ (!((neg_pl[pos] >> g) & 1)) {
                        b->hmask[l] |= (uint8_t) (1 << (c * 4 + j));
                    }
                }

                ਲਈ (ਪੂਰਨ sub = 0; sub < 2; sub++) {
                    sub_scales[kt_local * 2 + sub] = sc_pl[sub * 32 + ਪੰਕਤੀ];
                }
            }

            ਪੂਰਨ ls[16];
            ਜੇ (!hexagon_recover_k_scales(sub_scales, 16, -32, 31, &b->d, ls)) {
                ਦਸ਼ਮਲਵ max_scale = 0.0f;
                ਲਈ (ਪੂਰਨ s = 0; s < 16; s++) {
                    ਜੇ (fabsf(GGML_FP16_ਮੰਜ਼ਿਲ_FP32(sub_scales[s])) > fabsf(max_scale)) {
                        max_scale = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(sub_scales[s]);
                    }
                }
                b->d = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(-max_scale / 32.0f);
                ਅਚਲ ਦਸ਼ਮਲਵ d_actual = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(b->d);
                ਅਚਲ ਦਸ਼ਮਲਵ inv_d    = (d_actual != 0.0f) ? (1.0f / d_actual) : 0.0f;
                ਲਈ (ਪੂਰਨ s = 0; s < 16; s++) {
                    ls[s] = (std::max)(-32, (std::min)(31, (ਪੂਰਨ) roundf(GGML_FP16_ਮੰਜ਼ਿਲ_FP32(sub_scales[s]) * inv_d)));
                }
            }

            ਲਈ (ਪੂਰਨ s = 0; s < 16; s++) {
                ਅਚਲ ਪੂਰਨ l = ls[s] + 32;
                ਜੇ (s < 8) {
                    b->scales[s] = l & 0xF;
                } ਨਹੀਂਤਾਂ {
                    b->scales[s - 8] |= (uint8_t) ((l & 0xF) << 4);
                }
                b->scales[s % 4 + 8] |= (uint8_t) ((l >> 4) << (2 * (s / 4)));
            }
        });
}

ਸਥਿਰ ਖਾਲੀ repack_q2_K_tiled(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    repack_kquant_ਮੰਜ਼ਿਲ_tiled_grid<ਖੰਡ_q2_K>(t, ਅੰਕੜੇ, ਵਿੱਥ, size, HTP_TENSOR_TILE_ਮਾਪ_Q2_K,
        [](uint8_t * tile, ਅਚਲ ਖੰਡ_q2_K * b, ਪੂਰਨ ਪੰਕਤੀ, ਪੂਰਨ kt_local) {
            uint8_t * lo_pl   = tile;
            uint8_t * sc_pl   = tile + 256;
            ggml_half * dd_pl = (ggml_half *) (tile + 320);

#ਜੇ defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q2_3_k_low2(b->qs, kt_local, lo_pl, ਪੰਕਤੀ);
#ਨਹੀਂਤਾਂ
            ਲਈ (ਪੂਰਨ lk = 0; lk < 32; lk++) {
                ਅਚਲ ਪੂਰਨ g   = lk >> 2;
                ਅਚਲ ਪੂਰਨ pos = ਪੰਕਤੀ * 4 + (lk & 3);
                lo_pl[(g >> 2) * 128 + pos] |= (uint8_t) (q2_3_K_get_low2(b->qs, kt_local * 32 + lk) << ((g & 3) * 2));
            }
#ਜੇ_ਅੰਤ
            ਲਈ (ਪੂਰਨ sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + ਪੰਕਤੀ] = b->scales[kt_local * 2 + sub];
            }
            ਜੇ (ਪੰਕਤੀ / 8 == kt_local % 4) {
                dd_pl[ਪੰਕਤੀ % 8] = kt_local < 4 ? b->d : (ggml_half) (b->dmin ^ 0x8000);
            }
        });
}

ਸਥਿਰ ਖਾਲੀ repack_tiled_q2_K(ਖਾਲੀ * ਅੰਕੜੇ, ਅਚਲ ggml_tensor * t, size_t ਵਿੱਥ, size_t size) {
    unpack_kquant_from_tiled_grid<ਖੰਡ_q2_K>(ਅੰਕੜੇ, t, ਵਿੱਥ, size, HTP_TENSOR_TILE_ਮਾਪ_Q2_K,
        [](ਖੰਡ_q2_K * b, ਅਚਲ uint8_t * sb_tiles, size_t tile_ਮਾਪ, ਪੂਰਨ ਪੰਕਤੀ) {
            ਲਈ (ਪੂਰਨ kt_local = 0; kt_local < 8; kt_local++) {
                ਅਚਲ uint8_t *   tile  = sb_tiles + kt_local * tile_ਮਾਪ;
                ਅਚਲ uint8_t *   lo_pl = tile;
                ਅਚਲ uint8_t *   sc_pl = tile + 256;
                ਅਚਲ ggml_half * dd_pl = (ਅਚਲ ggml_half *) (tile + 320);

                ਲਈ (ਪੂਰਨ lk = 0; lk < 32; lk++) {
                    ਅਚਲ ਪੂਰਨ e   = kt_local * 32 + lk;
                    ਅਚਲ ਪੂਰਨ g   = lk >> 2;
                    ਅਚਲ ਪੂਰਨ pos = ਪੰਕਤੀ * 4 + (lk & 3);
                    ਅਚਲ uint8_t lo = (lo_pl[(g >> 2) * 128 + pos] >> ((g & 3) * 2)) & 3;
                    b->qs[(e / 128) * 32 + e % 32] |= (uint8_t) (lo << (2 * ((e % 128) / 32)));
                }

                ਲਈ (ਪੂਰਨ sub = 0; sub < 2; sub++) {
                    b->scales[kt_local * 2 + sub] = sc_pl[sub * 32 + ਪੰਕਤੀ];
                }
                ਜੇ (ਪੰਕਤੀ / 8 == kt_local % 4) {
                    ਜੇ (kt_local < 4) {
                        b->d = dd_pl[ਪੰਕਤੀ % 8];
                    } ਨਹੀਂਤਾਂ {
                        b->dmin = (ggml_half) (dd_pl[ਪੰਕਤੀ % 8] ^ 0x8000);
                    }
                }
            }
        });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ get_scale_min_k4(ਪੂਰਨ j, ਅਚਲ uint8_t * q, uint8_t * d, uint8_t * m) {
    ਜੇ (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } ਨਹੀਂਤਾਂ {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ pack_k4_scales_mins(uint8_t * scales, ggml_half * d, ggml_half * dmin,
                                       ਅਚਲ ਦਸ਼ਮਲਵ * sub_scales, ਅਚਲ ਦਸ਼ਮਲਵ * sub_mins) {
    ਦਸ਼ਮਲਵ max_scale = 0.0f;
    ਦਸ਼ਮਲਵ max_min   = 0.0f;
    ਲਈ (ਪੂਰਨ j = 0; j < 8; j++) {
        ਜੇ (sub_scales[j] > max_scale) max_scale = sub_scales[j];
        ਜੇ (sub_mins[j]   > max_min)   max_min   = sub_mins[j];
    }

    ਦਸ਼ਮਲਵ inv_scale = 0.0f;
    ਜੇ (max_scale > 0.0f) {
        *d = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(max_scale / 63.0f);
        ਅਚਲ ਦਸ਼ਮਲਵ d_actual = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(*d);
        inv_scale = (d_actual > 0.0f) ? (1.0f / d_actual) : 0.0f;
    } ਨਹੀਂਤਾਂ {
        *d = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(0.0f);
    }

    ਦਸ਼ਮਲਵ inv_min = 0.0f;
    ਜੇ (max_min > 0.0f) {
        *dmin = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(max_min / 63.0f);
        ਅਚਲ ਦਸ਼ਮਲਵ dmin_actual = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(*dmin);
        inv_min = (dmin_actual > 0.0f) ? (1.0f / dmin_actual) : 0.0f;
    } ਨਹੀਂਤਾਂ {
        *dmin = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(0.0f);
    }

    ਲਈ (ਪੂਰਨ j = 0; j < 8; j++) {
        uint8_t ls = (std::min)((uint8_t) 63, (uint8_t) roundf(inv_scale * sub_scales[j]));
        uint8_t lm = (std::min)((uint8_t) 63, (uint8_t) roundf(inv_min * sub_mins[j]));
        ਜੇ (j < 4) {
            scales[j]     = ls;
            scales[j + 4] = lm;
        } ਨਹੀਂਤਾਂ {
            scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            scales[j - 4] |= ((ls >> 4) << 6);
            scales[j - 0] |= ((lm >> 4) << 6);
        }
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ ਖੰਡ_T>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ repack_k4_quants(uint8_t * tile_dst, ਅਚਲ ਖੰਡ_T * b, ਪੂਰਨ ਪੰਕਤੀ, ਪੂਰਨ kt_local,
                                    int64_t sb_ਪ੍ਰਤੀ_ਪੰਕਤੀ, ਤਾਰਕਿਕ use_neon) {
#ਜੇ defined(__ARM_NEON) || defined(__aarch64__)
    ਜੇ (use_neon) {
        ਜੇ ((ਪੰਕਤੀ & 7) == 0) {
            ਅਚਲ uint8_t * qs[8];
            ਲਈ (ਪੂਰਨ rr = 0; rr < 8; ++rr) {
                qs[rr] = b[rr * sb_ਪ੍ਰਤੀ_ਪੰਕਤੀ].qs + (kt_local / 2) * 32;
            }
            repack_neon_k4_ਪੰਕਤੀਆਂ(qs, tile_dst, ਪੰਕਤੀ, (kt_local & 1) != 0);
        }
        ਵਾਪਸ;
    }
#ਨਹੀਂਤਾਂ
    GGML_ਅਣਵਰਤਿਆ(sb_ਪ੍ਰਤੀ_ਪੰਕਤੀ);
    GGML_ਅਣਵਰਤਿਆ(use_neon);
#ਜੇ_ਅੰਤ
    ਅਚਲ uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
    ਅਚਲ ਪੂਰਨ ਖਿਸਕਾਓ = (kt_local & 1) ? 4 : 0;

    ਲਈ (ਪੂਰਨ cp = 0; cp < 16; cp++) {
        ਅਚਲ uint8_t q0 = (qs_sub[2 * cp + 0] >> ਖਿਸਕਾਓ) & 0x0F;
        ਅਚਲ uint8_t q1 = (qs_sub[2 * cp + 1] >> ਖਿਸਕਾਓ) & 0x0F;
        tile_dst[cp * 32 + ਪੰਕਤੀ] = (uint8_t) ((q1 << 4) | q0);
    }
}

ਸਥਿਰ ਖਾਲੀ repack_q4_K_tiled(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    ਅਚਲ int64_t sb_ਪ੍ਰਤੀ_ਪੰਕਤੀ = t->ne[0] / QK_K;
    ਅਚਲ ਤਾਰਕਿਕ    use_neon   = (t->ne[1] % 8 == 0);
    repack_kquant_ਮੰਜ਼ਿਲ_tiled_grid<ਖੰਡ_q4_K>(t, ਅੰਕੜੇ, ਵਿੱਥ, size, HTP_TENSOR_TILE_ਮਾਪ_Q4_K,
        [sb_ਪ੍ਰਤੀ_ਪੰਕਤੀ, use_neon](uint8_t * tile_dst, ਅਚਲ ਖੰਡ_q4_K * b, ਪੂਰਨ ਪੰਕਤੀ, ਪੂਰਨ kt_local) {
            ਅਚਲ ਦਸ਼ਮਲਵ d    = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(b->d);
            ਅਚਲ ਦਸ਼ਮਲਵ dmin = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(b->dmin);

            uint8_t sc, m;
            get_scale_min_k4(kt_local, b->scales, &sc, &m);

            repack_k4_quants(tile_dst, b, ਪੰਕਤੀ, kt_local, sb_ਪ੍ਰਤੀ_ਪੰਕਤੀ, use_neon);

            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
            scale_dst[2 * ਪੰਕਤੀ + 0] = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(d * (ਦਸ਼ਮਲਵ) sc);
            scale_dst[2 * ਪੰਕਤੀ + 1] = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(-dmin * (ਦਸ਼ਮਲਵ) m);
        });
}

ਸਥਿਰ ਖਾਲੀ repack_tiled_q4_K(ਖਾਲੀ * ਅੰਕੜੇ, ਅਚਲ ggml_tensor * t, size_t ਵਿੱਥ, size_t size) {
    unpack_kquant_from_tiled_grid<ਖੰਡ_q4_K>(ਅੰਕੜੇ, t, ਵਿੱਥ, size, HTP_TENSOR_TILE_ਮਾਪ_Q4_K,
        [](ਖੰਡ_q4_K * b, ਅਚਲ uint8_t * sb_tiles, size_t tile_ਮਾਪ, ਪੂਰਨ ਪੰਕਤੀ) {
            ਦਸ਼ਮਲਵ sub_scales[8];
            ਦਸ਼ਮਲਵ sub_mins[8];

            ਲਈ (ਪੂਰਨ kt_local = 0; kt_local < 8; kt_local++) {
                ਅਚਲ uint8_t *   tile_src  = sb_tiles + kt_local * tile_ਮਾਪ;
                ਅਚਲ ggml_half * scale_src = (ਅਚਲ ggml_half *) (tile_src + 512);

                uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
                ਅਚਲ ਪੂਰਨ ਖਿਸਕਾਓ  = (kt_local & 1) ? 4 : 0;

                ਲਈ (ਪੂਰਨ cp = 0; cp < 16; cp++) {
                    ਅਚਲ uint8_t val = tile_src[cp * 32 + ਪੰਕਤੀ];
                    qs_sub[2 * cp + 0] |= (uint8_t) ((val & 0x0F) << ਖਿਸਕਾਓ);
                    qs_sub[2 * cp + 1] |= (uint8_t) ((val >> 4)   << ਖਿਸਕਾਓ);
                }

                ਅਚਲ ਦਸ਼ਮਲਵ D = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(scale_src[2 * ਪੰਕਤੀ + 0]);
                ਅਚਲ ਦਸ਼ਮਲਵ M = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(scale_src[2 * ਪੰਕਤੀ + 1]);
                sub_scales[kt_local] = (D > 0.0f) ? D : 0.0f;
                sub_mins[kt_local]   = (-M > 0.0f) ? -M : 0.0f;
            }

            pack_k4_scales_mins(b->scales, &b->d, &b->dmin, sub_scales, sub_mins);
        });
}

ਸਥਿਰ ਖਾਲੀ repack_q5_K_tiled(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    ਅਚਲ int64_t sb_ਪ੍ਰਤੀ_ਪੰਕਤੀ = t->ne[0] / QK_K;
    ਅਚਲ ਤਾਰਕਿਕ    use_neon   = (t->ne[1] % 8 == 0);
    repack_kquant_ਮੰਜ਼ਿਲ_tiled_grid<ਖੰਡ_q5_K>(t, ਅੰਕੜੇ, ਵਿੱਥ, size, HTP_TENSOR_TILE_ਮਾਪ_Q5_K,
        [sb_ਪ੍ਰਤੀ_ਪੰਕਤੀ, use_neon](uint8_t * tile_dst, ਅਚਲ ਖੰਡ_q5_K * b, ਪੂਰਨ ਪੰਕਤੀ, ਪੂਰਨ kt_local) {
            ਅਚਲ ਦਸ਼ਮਲਵ d     = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(b->d);
            ਅਚਲ ਦਸ਼ਮਲਵ dmin  = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(b->dmin);
            uint8_t * plane   = tile_dst + 640;

            uint8_t sc, m;
            get_scale_min_k4(kt_local, b->scales, &sc, &m);

            repack_k4_quants(tile_dst, b, ਪੰਕਤੀ, kt_local, sb_ਪ੍ਰਤੀ_ਪੰਕਤੀ, use_neon);

            ਅਚਲ uint8_t hbit = (uint8_t) (1 << kt_local);
#ਜੇ defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q5_k_qh(b->qh, hbit, plane, ਪੰਕਤੀ);
#ਨਹੀਂਤਾਂ
            ਲਈ (ਪੂਰਨ cp = 0; cp < 16; cp++) {
                ਅਚਲ ਪੂਰਨ i    = cp / 4;
                ਅਚਲ ਪੂਰਨ lane = (cp % 4) * 32 + ਪੰਕਤੀ;
                ਜੇ (b->qh[2 * cp + 0] & hbit) {
                    plane[lane] |= (uint8_t) (1 << (2 * i));
                }
                ਜੇ (b->qh[2 * cp + 1] & hbit) {
                    plane[lane] |= (uint8_t) (1 << (2 * i + 1));
                }
            }
#ਜੇ_ਅੰਤ

            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
            scale_dst[2 * ਪੰਕਤੀ + 0] = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(d * (ਦਸ਼ਮਲਵ) sc);
            scale_dst[2 * ਪੰਕਤੀ + 1] = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(-dmin * (ਦਸ਼ਮਲਵ) m);
        });
}

ਸਥਿਰ ਖਾਲੀ repack_tiled_q5_K(ਖਾਲੀ * ਅੰਕੜੇ, ਅਚਲ ggml_tensor * t, size_t ਵਿੱਥ, size_t size) {
    unpack_kquant_from_tiled_grid<ਖੰਡ_q5_K>(ਅੰਕੜੇ, t, ਵਿੱਥ, size, HTP_TENSOR_TILE_ਮਾਪ_Q5_K,
        [](ਖੰਡ_q5_K * b, ਅਚਲ uint8_t * sb_tiles, size_t tile_ਮਾਪ, ਪੂਰਨ ਪੰਕਤੀ) {
            ਦਸ਼ਮਲਵ sub_scales[8];
            ਦਸ਼ਮਲਵ sub_mins[8];

            ਲਈ (ਪੂਰਨ kt_local = 0; kt_local < 8; kt_local++) {
                ਅਚਲ uint8_t *   tile_src  = sb_tiles + kt_local * tile_ਮਾਪ;
                ਅਚਲ uint8_t *   plane     = tile_src + 640;
                ਅਚਲ ggml_half * scale_src = (ਅਚਲ ggml_half *) (tile_src + 512);

                uint8_t * qs_sub   = b->qs + (kt_local / 2) * 32;
                ਅਚਲ ਪੂਰਨ ਖਿਸਕਾਓ    = (kt_local & 1) ? 4 : 0;
                ਅਚਲ uint8_t hbit = (uint8_t) (1 << kt_local);

                ਲਈ (ਪੂਰਨ cp = 0; cp < 16; cp++) {
                    ਅਚਲ uint8_t val = tile_src[cp * 32 + ਪੰਕਤੀ];
                    qs_sub[2 * cp + 0] |= (uint8_t) ((val & 0x0F) << ਖਿਸਕਾਓ);
                    qs_sub[2 * cp + 1] |= (uint8_t) ((val >> 4)   << ਖਿਸਕਾਓ);

                    ਅਚਲ ਪੂਰਨ i    = cp / 4;
                    ਅਚਲ ਪੂਰਨ lane = (cp % 4) * 32 + ਪੰਕਤੀ;
                    ਜੇ (plane[lane] & (1 << (2 * i))) {
                        b->qh[2 * cp + 0] |= hbit;
                    }
                    ਜੇ (plane[lane] & (1 << (2 * i + 1))) {
                        b->qh[2 * cp + 1] |= hbit;
                    }
                }

                ਅਚਲ ਦਸ਼ਮਲਵ D = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(scale_src[2 * ਪੰਕਤੀ + 0]);
                ਅਚਲ ਦਸ਼ਮਲਵ M = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(scale_src[2 * ਪੰਕਤੀ + 1]);
                sub_scales[kt_local] = (D > 0.0f) ? D : 0.0f;
                sub_mins[kt_local]   = (-M > 0.0f) ? -M : 0.0f;
            }

            pack_k4_scales_mins(b->scales, &b->d, &b->dmin, sub_scales, sub_mins);
        });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ q1_0_tiled_ਦੋਆਧਾਰੀ_ਅੰਕ_location(ਪੂਰਨ k, ਪੂਰਨ ਪੰਕਤੀ, ਪੂਰਨ & byte_ਵਿੱਥ, ਪੂਰਨ & ਦੋਆਧਾਰੀ_ਅੰਕ) {
    ਅਚਲ ਪੂਰਨ quarter = k / 32;
    ਅਚਲ ਪੂਰਨ kq      = k % 32;

    byte_ਵਿੱਥ = quarter * 128 + 4 * ਪੰਕਤੀ + (kq % 4);
    ਦੋਆਧਾਰੀ_ਅੰਕ         = kq / 4;
}

ਸਥਿਰ ਖਾਲੀ repack_q1_0_tiled(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    ਅਚਲ ਖੰਡ_q1_0 * src_ਮੈਟ੍ਰਿਕਸ = (ਅਚਲ ਖੰਡ_q1_0 *) ਅੰਕੜੇ;
    ਅਚਲ int64_t ne0 = t->ne[0];
    ਅਚਲ int64_t ne1 = t->ne[1];
    ਅਚਲ int64_t ne2 = t->ne[2];
    ਅਚਲ int64_t ne3 = t->ne[3];
    ਅਚਲ ਪੂਰਨ n_ਖਾਨਾ_tiles = hex_ਗੇੜਾ_ਉੱਪਰ(ne1, 32) / 32;
    ਅਚਲ ਪੂਰਨ n_k_tiles = ne0 / QK1_0;
    ਅਚਲ size_t tile_ਮਾਪ = HTP_TENSOR_TILE_ਮਾਪ_Q1_0;
    ਅਚਲ size_t ਮੈਟ੍ਰਿਕਸ_ਮਾਪ = (size_t) n_ਖਾਨਾ_tiles * n_k_tiles * tile_ਮਾਪ;
    ਅਚਲ size_t ਟੁਕੜਾ_ਮਾਪ = ne1 * ggml_ਪੰਕਤੀ_ਮਾਪ(t->type, ne0);
    ਅਚਲ int64_t ਸ਼ੁਰੂ_ਟੁਕੜਾ = ਵਿੱਥ / ਟੁਕੜਾ_ਮਾਪ;
    ਅਚਲ int64_t ਸਿਰਾ_ਟੁਕੜਾ = (std::min)((int64_t) ((ਵਿੱਥ + size + ਟੁਕੜਾ_ਮਾਪ - 1) / ਟੁਕੜਾ_ਮਾਪ), ne2 * ne3);

    GGML_ASSERT(ne0 % QK1_0 == 0);

    ਲਈ (int64_t ਟੁਕੜਾ_idx = ਸ਼ੁਰੂ_ਟੁਕੜਾ; ਟੁਕੜਾ_idx < ਸਿਰਾ_ਟੁਕੜਾ; ਟੁਕੜਾ_idx++) {
        ਅਚਲ ਖੰਡ_q1_0 * src_ਟੁਕੜਾ = src_ਮੈਟ੍ਰਿਕਸ + (ਟੁਕੜਾ_idx - ਸ਼ੁਰੂ_ਟੁਕੜਾ) * ne1 * n_k_tiles;
        uint8_t * ਮੈਟ੍ਰਿਕਸ_dst = (uint8_t *) t->ਅੰਕੜੇ + ਟੁਕੜਾ_idx * ਮੈਟ੍ਰਿਕਸ_ਮਾਪ;

        repack_parallel_for_ct(ਮੈਟ੍ਰਿਕਸ_ਮਾਪ, n_ਖਾਨਾ_tiles, [&](ਪੂਰਨ ct0, ਪੂਰਨ ct1) {
            ਲਈ (ਪੂਰਨ ct = ct0; ct < ct1; ct++) {
                ਲਈ (ਪੂਰਨ kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = ਮੈਟ੍ਰਿਕਸ_dst + ((size_t) ct * n_k_tiles + kt) * tile_ਮਾਪ;
                    ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
                    memset(tile_dst, 0, tile_ਮਾਪ);

                    ਲਈ (ਪੂਰਨ ਪੰਕਤੀ = 0; ਪੰਕਤੀ < 32; ਪੰਕਤੀ++) {
                        ਅਚਲ int64_t r = ct * 32 + ਪੰਕਤੀ;
                        ਜੇ (r >= ne1) {
                            ਜਾਰੀ;
                        }

                        ਅਚਲ ਖੰਡ_q1_0 & ਖੰਡ = src_ਟੁਕੜਾ[r * n_k_tiles + kt];
                        scale_dst[ਪੰਕਤੀ] = ਖੰਡ.d;
#ਜੇ defined(__ARM_NEON) || defined(__aarch64__)
                        repack_neon_q1_0_ਖੰਡ(ਖੰਡ.qs, tile_dst, ਪੰਕਤੀ);
#ਨਹੀਂਤਾਂ
                        ਲਈ (ਪੂਰਨ k = 0; k < QK1_0; k++) {
                            ਪੂਰਨ byte_ਵਿੱਥ;
                            ਪੂਰਨ ਦੋਆਧਾਰੀ_ਅੰਕ;
                            q1_0_tiled_ਦੋਆਧਾਰੀ_ਅੰਕ_location(k, ਪੰਕਤੀ, byte_ਵਿੱਥ, ਦੋਆਧਾਰੀ_ਅੰਕ);
                            ਜੇ (ਖੰਡ.qs[k / 8] & (1u << (k % 8))) {
                                tile_dst[byte_ਵਿੱਥ] |= (uint8_t) (1u << ਦੋਆਧਾਰੀ_ਅੰਕ);
                            }
                        }
#ਜੇ_ਅੰਤ
                    }
                }
            }
        });
    }
}

ਸਥਿਰ ਖਾਲੀ repack_tiled_q1_0(ਖਾਲੀ * ਅੰਕੜੇ, ਅਚਲ ggml_tensor * t, size_t ਵਿੱਥ, size_t size) {
    ਖੰਡ_q1_0 * dst_ਮੈਟ੍ਰਿਕਸ = (ਖੰਡ_q1_0 *) ਅੰਕੜੇ;
    ਅਚਲ int64_t ne0 = t->ne[0];
    ਅਚਲ int64_t ne1 = t->ne[1];
    ਅਚਲ int64_t ne2 = t->ne[2];
    ਅਚਲ int64_t ne3 = t->ne[3];
    ਅਚਲ ਪੂਰਨ n_ਖਾਨਾ_tiles = hex_ਗੇੜਾ_ਉੱਪਰ(ne1, 32) / 32;
    ਅਚਲ ਪੂਰਨ n_k_tiles = ne0 / QK1_0;
    ਅਚਲ size_t tile_ਮਾਪ = HTP_TENSOR_TILE_ਮਾਪ_Q1_0;
    ਅਚਲ size_t ਮੈਟ੍ਰਿਕਸ_ਮਾਪ = (size_t) n_ਖਾਨਾ_tiles * n_k_tiles * tile_ਮਾਪ;
    ਅਚਲ size_t ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ = ggml_ਪੰਕਤੀ_ਮਾਪ(t->type, ne0);
    ਅਚਲ size_t ਟੁਕੜਾ_ਮਾਪ = ne1 * ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ;
    ਅਚਲ int64_t ਸ਼ੁਰੂ_ਟੁਕੜਾ = ਵਿੱਥ / ਟੁਕੜਾ_ਮਾਪ;
    ਅਚਲ int64_t ਸਿਰਾ_ਟੁਕੜਾ = (std::min)((int64_t) ((ਵਿੱਥ + size + ਟੁਕੜਾ_ਮਾਪ - 1) / ਟੁਕੜਾ_ਮਾਪ), ne2 * ne3);

    GGML_ASSERT(ne0 % QK1_0 == 0);

    ਲਈ (int64_t ਟੁਕੜਾ_idx = ਸ਼ੁਰੂ_ਟੁਕੜਾ; ਟੁਕੜਾ_idx < ਸਿਰਾ_ਟੁਕੜਾ; ਟੁਕੜਾ_idx++) {
        ਅਚਲ size_t cur_ਸ਼ੁਰੂ_byte = (std::max)(ਵਿੱਥ, (size_t) ਟੁਕੜਾ_idx * ਟੁਕੜਾ_ਮਾਪ);
        ਅਚਲ size_t cur_ਸਿਰਾ_byte = (std::min)(ਵਿੱਥ + size, (size_t) (ਟੁਕੜਾ_idx + 1) * ਟੁਕੜਾ_ਮਾਪ);
        ਅਚਲ int64_t ਸ਼ੁਰੂ_ਪੰਕਤੀ = (cur_ਸ਼ੁਰੂ_byte - (size_t) ਟੁਕੜਾ_idx * ਟੁਕੜਾ_ਮਾਪ) / ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ;
        ਅਚਲ int64_t ਸਿਰਾ_ਪੰਕਤੀ = (std::min)((int64_t) ((cur_ਸਿਰਾ_byte - (size_t) ਟੁਕੜਾ_idx * ਟੁਕੜਾ_ਮਾਪ + ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ - 1) / ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ), ne1);
        ਖੰਡ_q1_0 * dst_ਟੁਕੜਾ = dst_ਮੈਟ੍ਰਿਕਸ + (cur_ਸ਼ੁਰੂ_byte - ਵਿੱਥ) / ਆਕਾਰ(ਖੰਡ_q1_0);
        ਅਚਲ uint8_t * ਮੈਟ੍ਰਿਕਸ_src = (ਅਚਲ uint8_t *) t->ਅੰਕੜੇ + ਟੁਕੜਾ_idx * ਮੈਟ੍ਰਿਕਸ_ਮਾਪ;

        ਲਈ (ਪੂਰਨ ct = ਸ਼ੁਰੂ_ਪੰਕਤੀ / 32; ct < (ਸਿਰਾ_ਪੰਕਤੀ + 31) / 32; ct++) {
            ਲਈ (ਪੂਰਨ kt = 0; kt < n_k_tiles; kt++) {
                ਅਚਲ uint8_t * tile_src = ਮੈਟ੍ਰਿਕਸ_src + ((size_t) ct * n_k_tiles + kt) * tile_ਮਾਪ;
                ਅਚਲ ggml_half * scale_src = (ਅਚਲ ggml_half *) (tile_src + 512);

                ਲਈ (ਪੂਰਨ ਪੰਕਤੀ = 0; ਪੰਕਤੀ < 32; ਪੰਕਤੀ++) {
                    ਅਚਲ int64_t r = ct * 32 + ਪੰਕਤੀ;
                    ਜੇ (r < ਸ਼ੁਰੂ_ਪੰਕਤੀ || r >= ਸਿਰਾ_ਪੰਕਤੀ) {
                        ਜਾਰੀ;
                    }

                    ਖੰਡ_q1_0 & ਖੰਡ = dst_ਟੁਕੜਾ[(r - ਸ਼ੁਰੂ_ਪੰਕਤੀ) * n_k_tiles + kt];
                    memset(ਖੰਡ.qs, 0, ਆਕਾਰ(ਖੰਡ.qs));
                    ਖੰਡ.d = scale_src[ਪੰਕਤੀ];
                    ਲਈ (ਪੂਰਨ k = 0; k < QK1_0; k++) {
                        ਪੂਰਨ byte_ਵਿੱਥ;
                        ਪੂਰਨ ਦੋਆਧਾਰੀ_ਅੰਕ;
                        q1_0_tiled_ਦੋਆਧਾਰੀ_ਅੰਕ_location(k, ਪੰਕਤੀ, byte_ਵਿੱਥ, ਦੋਆਧਾਰੀ_ਅੰਕ);
                        ਜੇ (tile_src[byte_ਵਿੱਥ] & (1u << ਦੋਆਧਾਰੀ_ਅੰਕ)) {
                            ਖੰਡ.qs[k / 8] |= (uint8_t) (1u << (k % 8));
                        }
                    }
                }
            }
        }
    }
}

// --- Top-level Repack / Unpack Dispatchers ---

ਖਾਲੀ htp_repack_tensor_tiled(ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t size) {
    ਚੋਣ (tensor->type) {
        ਹਾਲਤ GGML_TYPE_Q1_0:
            repack_q1_0_tiled(tensor, ਅੰਕੜੇ, 0, size);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
            repack_q4_0_tiled(tensor, ਅੰਕੜੇ, 0, size);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_1:
            repack_q4_1_tiled(tensor, ਅੰਕੜੇ, 0, size);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q8_0:
            repack_q8_0_tiled(tensor, ਅੰਕੜੇ, 0, size);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_MXFP4:
            repack_mxfp4_tiled(tensor, ਅੰਕੜੇ, 0, size);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q2_0:
            repack_q2_0_tiled(tensor, ਅੰਕੜੇ, 0, size);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q2_K:
            repack_q2_K_tiled(tensor, ਅੰਕੜੇ, 0, size);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q3_K:
            repack_q3_K_tiled(tensor, ਅੰਕੜੇ, 0, size);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_K:
            repack_q4_K_tiled(tensor, ਅੰਕੜੇ, 0, size);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_K:
            repack_q5_K_tiled(tensor, ਅੰਕੜੇ, 0, size);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q6_K:
            repack_q6_K_tiled(tensor, ਅੰਕੜੇ, 0, size);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            repack_float_ਮੰਜ਼ਿਲ_f16_tiled<uint16_t>(tensor, ਅੰਕੜੇ, 0, size,
                [](uint16_t v) { ਵਾਪਸ v; });
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F32:
            repack_float_ਮੰਜ਼ਿਲ_f16_tiled<ਦਸ਼ਮਲਵ>(tensor, ਅੰਕੜੇ, 0, size,
                [](ਦਸ਼ਮਲਵ v) { ਵਾਪਸ (uint16_t) GGML_FP32_ਮੰਜ਼ਿਲ_FP16(v); });
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_BF16:
            repack_float_ਮੰਜ਼ਿਲ_f16_tiled<ggml_bf16_t>(tensor, ਅੰਕੜੇ, 0, size,
                [](ggml_bf16_t v) { ਵਾਪਸ (uint16_t) GGML_FP32_ਮੰਜ਼ਿਲ_FP16(GGML_BF16_ਮੰਜ਼ਿਲ_FP32(v)); });
            ਰੋਕੋ;
        ਮੂਲ:
            ਰੋਕੋ;
    }
}

ਖਾਲੀ htp_unpack_tensor_tiled(ਖਾਲੀ * ਅੰਕੜੇ, ਅਚਲ ggml_tensor * tensor, size_t ਵਿੱਥ, size_t size) {
    GGML_ASSERT(ਵਿੱਥ + size <= ggml_nbytes(tensor));

    ਚੋਣ (tensor->type) {
        ਹਾਲਤ GGML_TYPE_Q1_0:
            repack_tiled_q1_0(ਅੰਕੜੇ, tensor, ਵਿੱਥ, size);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
            repack_tiled_q4_0(ਅੰਕੜੇ, tensor, ਵਿੱਥ, size);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_1:
            repack_tiled_q4_1(ਅੰਕੜੇ, tensor, ਵਿੱਥ, size);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q8_0:
            repack_tiled_q8_0(ਅੰਕੜੇ, tensor, ਵਿੱਥ, size);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_MXFP4:
            repack_tiled_mxfp4(ਅੰਕੜੇ, tensor, ਵਿੱਥ, size);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q2_0:
            repack_tiled_q2_0(ਅੰਕੜੇ, tensor, ਵਿੱਥ, size);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q2_K:
            repack_tiled_q2_K(ਅੰਕੜੇ, tensor, ਵਿੱਥ, size);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q3_K:
            repack_tiled_q3_K(ਅੰਕੜੇ, tensor, ਵਿੱਥ, size);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_K:
            repack_tiled_q4_K(ਅੰਕੜੇ, tensor, ਵਿੱਥ, size);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_K:
            repack_tiled_q5_K(ਅੰਕੜੇ, tensor, ਵਿੱਥ, size);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q6_K:
            repack_tiled_q6_K(ਅੰਕੜੇ, tensor, ਵਿੱਥ, size);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            unpack_f16_tiled_ਮੰਜ਼ਿਲ_float<uint16_t>(ਅੰਕੜੇ, tensor, ਵਿੱਥ, size,
                [](uint16_t v) { ਵਾਪਸ v; });
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F32:
            unpack_f16_tiled_ਮੰਜ਼ਿਲ_float<ਦਸ਼ਮਲਵ>(ਅੰਕੜੇ, tensor, ਵਿੱਥ, size,
                [](uint16_t v) { ਵਾਪਸ ggml_fp16_ਮੰਜ਼ਿਲ_fp32((ggml_fp16_t) v); });
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_BF16:
            unpack_f16_tiled_ਮੰਜ਼ਿਲ_float<ggml_bf16_t>(ਅੰਕੜੇ, tensor, ਵਿੱਥ, size,
                [](uint16_t v) { ਵਾਪਸ ggml_fp32_ਮੰਜ਼ਿਲ_bf16(ggml_fp16_ਮੰਜ਼ਿਲ_fp32((ggml_fp16_t) v)); });
            ਰੋਕੋ;
        ਮੂਲ:
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਅੰਕੜੇ, (ਅਚਲ ਅੱਖਰ *) tensor->ਅੰਕੜੇ + ਵਿੱਥ, size);
            ਰੋਕੋ;
    }
}
