#ଅନ୍ତର୍ଭାଵ "htp-repack.h"

#define GGML_COMMON_IMPL_CPP
#ଅନ୍ତର୍ଭାଵ "ggml-common.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-quants.h"

#ଅନ୍ତର୍ଭାଵ <algorithm>
#ଅନ୍ତର୍ଭାଵ <cmath>
#ଅନ୍ତର୍ଭାଵ <cstring>
#ଅନ୍ତର୍ଭାଵ <initializer_list>
#ଅନ୍ତର୍ଭାଵ <thread>

#ଯଦି defined(__ARM_NEON) || defined(__aarch64__)
#    include <arm_neon.h>
#endif

// --- 4-bit block-32 nibble helpers (shared by Q4_0, IQ4_NL, Q4_1, MXFP4) ---

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ unpack_nibbles_32(uint8_t * dst, ଅଚଳ uint8_t * qs) {
    ପାଇଁ (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 16; ++i) {
        dst[i + 0]  = qs[i] & 0x0F;
        dst[i + 16] = qs[i] >> 4;
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ pack_nibbles_32(uint8_t * qs, ଅଚଳ uint8_t * src) {
    ପାଇଁ (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 16; ++i) {
        qs[i] = src[i + 0] | (src[i + 16] << 4);
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ transpose_nibbles_to_tile(uint8_t * tile_dst, ଅଚଳ uint8_t tile_quants[32][32]) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା cp = 0; cp < 16; cp++) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = 0; row < 32; row++) {
            tile_dst[cp * 32 + row] = (tile_quants[row][2 * cp + 1] << 4) | tile_quants[row][2 * cp];
        }
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ transpose_tile_to_nibbles(uint8_t tile_quants[32][32], ଅଚଳ uint8_t * tile_src) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା cp = 0; cp < 16; cp++) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = 0; row < 32; row++) {
            ଅଚଳ uint8_t val = tile_src[cp * 32 + row];
            tile_quants[row][2 * cp + 0] = val & 0x0F;
            tile_quants[row][2 * cp + 1] = val >> 4;
        }
    }
}

#ଯଦି defined(__ARM_NEON) || defined(__aarch64__)
ସ୍ଥିର ଶୂନ୍ୟ repack_neon_transpose8x8(ଅଚଳ uint8x8_t rows[8], uint8x8_t columns[8]) {
    ଅଚଳ uint8x8x2_t t0 = vtrn_u8(rows[0], rows[1]);
    ଅଚଳ uint8x8x2_t t1 = vtrn_u8(rows[2], rows[3]);
    ଅଚଳ uint8x8x2_t t2 = vtrn_u8(rows[4], rows[5]);
    ଅଚଳ uint8x8x2_t t3 = vtrn_u8(rows[6], rows[7]);
    ଅଚଳ uint16x4x2_t t4 = vtrn_u16(vreinterpret_u16_u8(t0.val[0]), vreinterpret_u16_u8(t1.val[0]));
    ଅଚଳ uint16x4x2_t t5 = vtrn_u16(vreinterpret_u16_u8(t0.val[1]), vreinterpret_u16_u8(t1.val[1]));
    ଅଚଳ uint16x4x2_t t6 = vtrn_u16(vreinterpret_u16_u8(t2.val[0]), vreinterpret_u16_u8(t3.val[0]));
    ଅଚଳ uint16x4x2_t t7 = vtrn_u16(vreinterpret_u16_u8(t2.val[1]), vreinterpret_u16_u8(t3.val[1]));
    ଅଚଳ uint32x2x2_t t8 = vtrn_u32(vreinterpret_u32_u16(t4.val[0]), vreinterpret_u32_u16(t6.val[0]));
    ଅଚଳ uint32x2x2_t t9 = vtrn_u32(vreinterpret_u32_u16(t4.val[1]), vreinterpret_u32_u16(t6.val[1]));
    ଅଚଳ uint32x2x2_t ta = vtrn_u32(vreinterpret_u32_u16(t5.val[0]), vreinterpret_u32_u16(t7.val[0]));
    ଅଚଳ uint32x2x2_t tb = vtrn_u32(vreinterpret_u32_u16(t5.val[1]), vreinterpret_u32_u16(t7.val[1]));
    columns[0] = vreinterpret_u8_u32(t8.val[0]);
    columns[1] = vreinterpret_u8_u32(ta.val[0]);
    columns[2] = vreinterpret_u8_u32(t9.val[0]);
    columns[3] = vreinterpret_u8_u32(tb.val[0]);
    columns[4] = vreinterpret_u8_u32(t8.val[1]);
    columns[5] = vreinterpret_u8_u32(ta.val[1]);
    columns[6] = vreinterpret_u8_u32(t9.val[1]);
    columns[7] = vreinterpret_u8_u32(tb.val[1]);
}

ସ୍ଥିର uint8x8_t repack_neon_pack_4bit_pairs(uint8x8_t first, uint8x8_t second, ତାର୍କିକ ଉପର) {
    ଅଚଳ uint8x8x2_t separated = vuzp_u8(first, second);
    ଅଚଳ uint8x8_t mask = vdup_n_u8(0x0F);
    uint8x8_t even = separated.val[0];
    uint8x8_t odd  = separated.val[1];

    ଯଦି (ଉପର) {
        even = vshr_n_u8(even, 4);
        odd  = vshr_n_u8(odd, 4);
    }

    even = vand_u8(even, mask);
    odd  = vand_u8(odd, mask);
    ଫେରନ୍ତୁ vorr_u8(vshl_n_u8(odd, 4), even);
}

ସ୍ଥିର ଶୂନ୍ୟ repack_neon_4bit_rows(ଅଚଳ uint8_t * qs[8], uint8_t * tile_dst, ପୂର୍ଣ୍ଣସଂଖ୍ୟା row0) {
    uint8x8_t ତଳ_rows[8];
    uint8x8_t ଉପର_rows[8];
    uint8x8_t columns[8];

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା rr = 0; rr < 8; ++rr) {
        ଅଚଳ uint8x16_t packed = vld1q_u8(qs[rr]);
        ଅଚଳ uint8x8_t first = vget_ତଳ_u8(packed);
        ଅଚଳ uint8x8_t second = vget_ଉପର_u8(packed);
        ତଳ_rows[rr]  = repack_neon_pack_4bit_pairs(first, second, ମିଥ୍ୟା);
        ଉପର_rows[rr] = repack_neon_pack_4bit_pairs(first, second, ସତ୍ୟ);
    }

    repack_neon_transpose8x8(ତଳ_rows, columns);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + cp * 32 + row0, columns[cp]);
    }

    repack_neon_transpose8x8(ଉପର_rows, columns);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + (cp + 8) * 32 + row0, columns[cp]);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ repack_neon_k4_rows(ଅଚଳ uint8_t * qs[8], uint8_t * tile_dst, ପୂର୍ଣ୍ଣସଂଖ୍ୟା row0, ତାର୍କିକ ଉପର) {
    uint8x8_t first_rows[8];
    uint8x8_t second_rows[8];
    uint8x8_t columns[8];

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା rr = 0; rr < 8; ++rr) {
        first_rows[rr] = repack_neon_pack_4bit_pairs(vld1_u8(qs[rr]), vld1_u8(qs[rr] + 8), ଉପର);
        second_rows[rr] = repack_neon_pack_4bit_pairs(vld1_u8(qs[rr] + 16), vld1_u8(qs[rr] + 24), ଉପର);
    }

    repack_neon_transpose8x8(first_rows, columns);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + cp * 32 + row0, columns[cp]);
    }

    repack_neon_transpose8x8(second_rows, columns);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + (cp + 8) * 32 + row0, columns[cp]);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ repack_neon_q8_rows(ଅଚଳ uint8_t * qs[8], uint8_t * tile_dst, ପୂର୍ଣ୍ଣସଂଖ୍ୟା row0) {
    uint8x8_t rows[8];
    uint8x8_t columns[8];

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା cp = 0; cp < 16; cp += 4) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା rr = 0; rr < 8; ++rr) {
            rows[rr] = vld1_u8(qs[rr] + 2 * cp);
        }
        repack_neon_transpose8x8(rows, columns);

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା pair = 0; pair < 4; ++pair) {
            ଅଚଳ uint8x8x2_t interleaved = vzip_u8(columns[2 * pair], columns[2 * pair + 1]);
            uint8_t * dst = tile_dst + (cp + pair) * 64 + 2 * row0;
            vst1_u8(dst, interleaved.val[0]);
            vst1_u8(dst + 8, interleaved.val[1]);
        }
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ repack_store_u32(uint8_t * dst, uint32_t v) {
    memcpy(dst, &v, ଆକାର(v));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint32x4_t repack_neon_transpose_2bit_4x4(uint32x4_t w) {
    uint32x4_t t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 6)), vdupq_n_u32(0x00CC00CCu));
    w = veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 6)));
    t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 12)), vdupq_n_u32(0x0000F0F0u));
    ଫେରନ୍ତୁ veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 12)));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ repack_neon_q2_0_block(ଅଚଳ uint8_t * qs, uint8_t * tile, ପୂର୍ଣ୍ଣସଂଖ୍ୟା row) {
    ଅଚଳ uint32x4_t w = repack_neon_transpose_2bit_4x4(vreinterpretq_u32_u8(vld1q_u8(qs)));
    repack_store_u32(tile + 0 * 128 + 4 * row, vgetq_lane_u32(w, 0));
    repack_store_u32(tile + 1 * 128 + 4 * row, vgetq_lane_u32(w, 1));
    repack_store_u32(tile + 2 * 128 + 4 * row, vgetq_lane_u32(w, 2));
    repack_store_u32(tile + 3 * 128 + 4 * row, vgetq_lane_u32(w, 3));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ repack_neon_q1_0_block(ଅଚଳ uint8_t * qs, uint8_t * tile, ପୂର୍ଣ୍ଣସଂଖ୍ୟା row) {
    uint32x4_t w = vreinterpretq_u32_u8(vld1q_u8(qs));
    uint32x4_t t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 1)), vdupq_n_u32(0x22222222u));
    w = veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 1)));
    t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 3)), vdupq_n_u32(0x0A0A0A0Au));
    w = veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 3)));
    w = repack_neon_transpose_2bit_4x4(w);
    repack_store_u32(tile + 0 * 128 + 4 * row, vgetq_lane_u32(w, 0));
    repack_store_u32(tile + 1 * 128 + 4 * row, vgetq_lane_u32(w, 1));
    repack_store_u32(tile + 2 * 128 + 4 * row, vgetq_lane_u32(w, 2));
    repack_store_u32(tile + 3 * 128 + 4 * row, vgetq_lane_u32(w, 3));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ repack_neon_pack_2bit_plane(uint8x16_t v0, uint8x16_t v1, uint8_t * dst, ପୂର୍ଣ୍ଣସଂଖ୍ୟା row) {
    ଅଚଳ uint32x4x2_t z   = vuzpq_u32(vreinterpretq_u32_u8(v0), vreinterpretq_u32_u8(v1));
    ଅଚଳ uint32x4_t   p01 = vorrq_u32(z.val[0], vshlq_n_u32(z.val[1], 2));
    ଅଚଳ uint32x2x2_t u   = vuzp_u32(vget_ତଳ_u32(p01), vget_ଉପର_u32(p01));
    ଅଚଳ uint32x2_t   out = vorr_u32(u.val[0], vshl_n_u32(u.val[1], 4));
    repack_store_u32(dst + 0 * 128 + row * 4, vget_lane_u32(out, 0));
    repack_store_u32(dst + 1 * 128 + row * 4, vget_lane_u32(out, 1));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ repack_neon_q2_3_k_low2(ଅଚଳ uint8_t * qs, ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt_local, uint8_t * lo_pl, ପୂର୍ଣ୍ଣସଂଖ୍ୟା row) {
    ଅଚଳ uint8_t *  qs_sub = qs + (kt_local / 4) * 32;
    ଅଚଳ int8x16_t  shift  = vdupq_n_s8(-2 * (kt_local % 4));
    ଅଚଳ uint8x16_t mask   = vdupq_n_u8(0x03);
    ଅଚଳ uint8x16_t v0     = vandq_u8(vshlq_u8(vld1q_u8(qs_sub +  0), shift), mask);
    ଅଚଳ uint8x16_t v1     = vandq_u8(vshlq_u8(vld1q_u8(qs_sub + 16), shift), mask);
    repack_neon_pack_2bit_plane(v0, v1, lo_pl, row);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ repack_neon_q3_k_hmask(ଅଚଳ uint8_t * hmask, ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt_local, uint8_t * neg_pl, ପୂର୍ଣ୍ଣସଂଖ୍ୟା row) {
    ସ୍ଥିର ଅଚଳ uint32_t mask_ଶବ୍ଦଗୁଡ଼ିକ[4] = { 0x01010101u, 0x02020202u, 0x04040404u, 0x08080808u };
    ଅଚଳ uint8x16_t m0   = vreinterpretq_u8_u32(vld1q_u32(mask_ଶବ୍ଦଗୁଡ଼ିକ));
    ଅଚଳ uint8x16_t m1   = vshlq_n_u8(m0, 4);
    ଅଚଳ uint8x16_t hbit = vdupq_n_u8((uint8_t) (1u << kt_local));
    ଅଚଳ uint8x16_t b0   = vbicq_u8(m0, vtstq_u8(vld1q_u8(hmask +  0), hbit));
    ଅଚଳ uint8x16_t b1   = vbicq_u8(m1, vtstq_u8(vld1q_u8(hmask + 16), hbit));
    ଅଚଳ uint32x4_t b01  = vreinterpretq_u32_u8(vorrq_u8(b0, b1));
    ଅଚଳ uint32x2_t f0   = vorr_u32(vget_ତଳ_u32(b01), vget_ଉପର_u32(b01));
    ଅଚଳ uint32x2_t f1   = vpadd_u32(f0, f0);
    repack_store_u32(neg_pl + row * 4, vget_lane_u32(f1, 0));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ repack_neon_q6_k(ଅଚଳ block_q6_K * b, ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt_local, uint8_t * lo_pl, uint8_t * hi_pl, ପୂର୍ଣ୍ଣସଂଖ୍ୟା row) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା        c      = kt_local / 4;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା        g_sb   = kt_local % 4;
    ଅଚଳ uint8_t *  ql     = b->ql + c * 64 + (g_sb & 1) * 32;
    ଅଚଳ uint8_t *  qh     = b->qh + c * 32;
    ଅଚଳ int8x16_t  lshift = vdupq_n_s8((g_sb >= 2) ? -4 : 0);
    ଅଚଳ int8x16_t  hshift = vdupq_n_s8(-2 * g_sb);
    ଅଚଳ uint8x16_t mask4  = vdupq_n_u8(0x0F);
    ଅଚଳ uint8x16_t mask2  = vdupq_n_u8(0x03);

    ଅଚଳ uint8x16_t lo0  = vandq_u8(vshlq_u8(vld1q_u8(ql +  0), lshift), mask4);
    ଅଚଳ uint8x16_t lo1  = vandq_u8(vshlq_u8(vld1q_u8(ql + 16), lshift), mask4);
    ଅଚଳ uint32x4x2_t lz = vuzpq_u32(vreinterpretq_u32_u8(lo0), vreinterpretq_u32_u8(lo1));
    ଅଚଳ uint32x4_t lout = vorrq_u32(lz.val[0], vshlq_n_u32(lz.val[1], 4));
    repack_store_u32(lo_pl + 0 * 128 + row * 4, vgetq_lane_u32(lout, 0));
    repack_store_u32(lo_pl + 1 * 128 + row * 4, vgetq_lane_u32(lout, 1));
    repack_store_u32(lo_pl + 2 * 128 + row * 4, vgetq_lane_u32(lout, 2));
    repack_store_u32(lo_pl + 3 * 128 + row * 4, vgetq_lane_u32(lout, 3));

    ଅଚଳ uint8x16_t hi0 = vandq_u8(vshlq_u8(vld1q_u8(qh +  0), hshift), mask2);
    ଅଚଳ uint8x16_t hi1 = vandq_u8(vshlq_u8(vld1q_u8(qh + 16), hshift), mask2);
    repack_neon_pack_2bit_plane(hi0, hi1, hi_pl, row);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ repack_neon_q5_k_qh(ଅଚଳ uint8_t * qh, uint8_t hbit, uint8_t * plane, ପୂର୍ଣ୍ଣସଂଖ୍ୟା row) {
    ସ୍ଥିର ଅଚଳ uint8_t mask_bytes[16] = { 1, 2, 1, 2, 1, 2, 1, 2, 4, 8, 4, 8, 4, 8, 4, 8 };
    ଅଚଳ uint8x16_t m0   = vld1q_u8(mask_bytes);
    ଅଚଳ uint8x16_t m1   = vshlq_n_u8(m0, 4);
    ଅଚଳ uint8x16_t vbit = vdupq_n_u8(hbit);
    ଅଚଳ uint8x16_t b0   = vandq_u8(vtstq_u8(vld1q_u8(qh +  0), vbit), m0);
    ଅଚଳ uint8x16_t b1   = vandq_u8(vtstq_u8(vld1q_u8(qh + 16), vbit), m1);
    ଅଚଳ uint8x16_t b01  = vorrq_u8(b0, b1);
    ଅଚଳ uint8x8_t  fold = vorr_u8(vget_ତଳ_u8(b01), vget_ଉପର_u8(b01));
    ଅଚଳ uint8x8_t  out  = vpadd_u8(fold, fold);
    plane[0 * 32 + row] = vget_lane_u8(out, 0);
    plane[1 * 32 + row] = vget_lane_u8(out, 1);
    plane[2 * 32 + row] = vget_lane_u8(out, 2);
    plane[3 * 32 + row] = vget_lane_u8(out, 3);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint16x8_t repack_neon_load_f16x8(ଅଚଳ uint16_t * src) {
    ଫେରନ୍ତୁ vld1q_u16(src);
}

#ଯଦି defined(__aarch64__) || defined(__ARM_FP16_FORMAT_IEEE)
ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint16x8_t repack_neon_load_f16x8(ଅଚଳ ଦଶମିକ * src) {
    ଅଚଳ float16x4_t lo = vcvt_f16_f32(vld1q_f32(src));
    ଅଚଳ float16x4_t hi = vcvt_f16_f32(vld1q_f32(src + 4));
    ଫେରନ୍ତୁ vreinterpretq_u16_f16(vcombine_f16(lo, hi));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint16x8_t repack_neon_load_f16x8(ଅଚଳ ggml_bf16_t * src) {
    ଅଚଳ uint16x8_t  bf   = vld1q_u16((ଅଚଳ uint16_t *) src);
    ଅଚଳ float32x4_t f_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_ତଳ_u16(bf), 16));
    ଅଚଳ float32x4_t f_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_ଉପର_u16(bf), 16));
    ଅଚଳ float16x4_t lo   = vcvt_f16_f32(f_lo);
    ଅଚଳ float16x4_t hi   = vcvt_f16_f32(f_hi);
    ଫେରନ୍ତୁ vreinterpretq_u16_f16(vcombine_f16(lo, hi));
}
#ଅନ୍ୟଥା
ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint16x8_t repack_neon_load_f16x8(ଅଚଳ ଦଶମିକ * src) {
    uint16_t tmp[8];
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 8; ++i) {
        tmp[i] = GGML_FP32_TO_FP16(src[i]);
    }
    ଫେରନ୍ତୁ vld1q_u16(tmp);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint16x8_t repack_neon_load_f16x8(ଅଚଳ ggml_bf16_t * src) {
    uint16_t tmp[8];
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 8; ++i) {
        tmp[i] = GGML_FP32_TO_FP16(GGML_BF16_TO_FP32(src[i]));
    }
    ଫେରନ୍ତୁ vld1q_u16(tmp);
}
#endif

template <typename SrcT>
ସ୍ଥିର ଶୂନ୍ୟ repack_neon_f16_tile(uint16_t * tile_dst, ଅଚଳ SrcT * src_slice, int64_t row_stride, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ct, ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା row0 = 0; row0 < 32; row0 += 4) {
        ଅଚଳ SrcT * r0 = src_slice + (int64_t) (ct * 32 + row0 + 0) * row_stride + kt * 32;
        ଅଚଳ SrcT * r1 = src_slice + (int64_t) (ct * 32 + row0 + 1) * row_stride + kt * 32;
        ଅଚଳ SrcT * r2 = src_slice + (int64_t) (ct * 32 + row0 + 2) * row_stride + kt * 32;
        ଅଚଳ SrcT * r3 = src_slice + (int64_t) (ct * 32 + row0 + 3) * row_stride + kt * 32;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା cp = 0; cp < 16; cp += 4) {
            ଅଚଳ uint32x4_t v0 = vreinterpretq_u32_u16(repack_neon_load_f16x8(r0 + 2 * cp));
            ଅଚଳ uint32x4_t v1 = vreinterpretq_u32_u16(repack_neon_load_f16x8(r1 + 2 * cp));
            ଅଚଳ uint32x4_t v2 = vreinterpretq_u32_u16(repack_neon_load_f16x8(r2 + 2 * cp));
            ଅଚଳ uint32x4_t v3 = vreinterpretq_u32_u16(repack_neon_load_f16x8(r3 + 2 * cp));

            ଅଚଳ uint32x2x2_t t0 = vtrn_u32(vget_ତଳ_u32(v0),  vget_ତଳ_u32(v1));
            ଅଚଳ uint32x2x2_t t1 = vtrn_u32(vget_ଉପର_u32(v0), vget_ଉପର_u32(v1));
            ଅଚଳ uint32x2x2_t t2 = vtrn_u32(vget_ତଳ_u32(v2),  vget_ତଳ_u32(v3));
            ଅଚଳ uint32x2x2_t t3 = vtrn_u32(vget_ଉପର_u32(v2), vget_ଉପର_u32(v3));

            vst1q_u16(tile_dst + (cp + 0) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t0.val[0], t2.val[0])));
            vst1q_u16(tile_dst + (cp + 1) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t0.val[1], t2.val[1])));
            vst1q_u16(tile_dst + (cp + 2) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t1.val[0], t3.val[0])));
            vst1q_u16(tile_dst + (cp + 3) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t1.val[1], t3.val[1])));
        }
    }
}
#endif

// --- Generic 32x32 tile grid walkers for block-32 and float types ---

template <typename Fn>
ସ୍ଥିର ଶୂନ୍ୟ repack_parallel_for_ct(size_t slice_bytes, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_col_tiles, Fn && fn) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads = 1;
    ଯଦି (slice_bytes > 128 * 1024) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା nt : { 4, 3, 2 }) {
            ଯଦି (n_col_tiles % nt == 0) {
                n_threads = nt;
                ଭାଙ୍ଗ;
            }
        }
    }

    ଯଦି (n_threads <= 1) {
        fn(0, n_col_tiles);
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା chunk = n_col_tiles / n_threads;
    std::thread workers[3];
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_threads - 1; ++i) {
        workers[i] = std::thread([&fn, i, chunk]() {
            fn(i * chunk, (i + 1) * chunk);
        });
    }
    fn((n_threads - 1) * chunk, n_col_tiles);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_threads - 1; ++i) {
        workers[i].join();
    }
}

template <typename SrcElemT, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ELEMS_PER_KTILE, typename PackTileFn>
ସ୍ଥିର ଶୂନ୍ୟ repack_to_tiled_grid_32(ggml_tensor * t, ଅଚଳ ଶୂନ୍ୟ * data, size_t offset, size_t size,
                                    size_t tile_size, PackTileFn && pack_tile) {
    ଅଚଳ SrcElemT * src_matrix = (ଅଚଳ SrcElemT *) data;
    ଅଚଳ int64_t ne0 = t->ne[0];
    ଅଚଳ int64_t ne1 = t->ne[1];
    ଅଚଳ int64_t ne2 = t->ne[2];
    ଅଚଳ int64_t ne3 = t->ne[3];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_col_tiles = hex_round_up(ne1, 32) / 32;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_k_tiles   = hex_round_up(ne0, 32) / 32;
    ଅଚଳ size_t matrix_size = (size_t) n_col_tiles * n_k_tiles * tile_size;

    ଅଚଳ size_t slice_size = (size_t) ne1 * ggml_row_size(t->type, ne0);
    ଅଚଳ int64_t start_slice = offset / slice_size;
    int64_t end_slice = (offset + size + slice_size - 1) / slice_size;
    ଯଦି (end_slice > ne2 * ne3) {
        end_slice = ne2 * ne3;
    }

    ଅଚଳ int64_t row_stride = (ne0 / (32 / ELEMS_PER_KTILE));

    ପାଇଁ (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        ଅଚଳ SrcElemT * src_slice = src_matrix + (slice_idx - start_slice) * (ne1 * row_stride);
        uint8_t * matrix_dst = (uint8_t *) t->data + slice_idx * matrix_size;

        repack_parallel_for_ct(matrix_size, n_col_tiles, [&](ପୂର୍ଣ୍ଣସଂଖ୍ୟା ct0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ct1) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ct = ct0; ct < ct1; ct++) {
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = matrix_dst + ((size_t) ct * n_k_tiles + kt) * tile_size;
                    pack_tile(tile_dst, src_slice, ne0, ne1, row_stride, ct, kt);
                }
            }
        });
    }
}

template <typename DstElemT, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ELEMS_PER_KTILE, typename UnpackTileFn>
ସ୍ଥିର ଶୂନ୍ୟ unpack_from_tiled_grid_32(ଶୂନ୍ୟ * data, ଅଚଳ ggml_tensor * t, size_t offset, size_t size,
                                      size_t tile_size, UnpackTileFn && unpack_tile) {
    uint8_t * dst_bytes = (uint8_t *) data;
    ଅଚଳ int64_t ne0 = t->ne[0];
    ଅଚଳ int64_t ne1 = t->ne[1];
    ଅଚଳ int64_t ne2 = t->ne[2];
    ଅଚଳ int64_t ne3 = t->ne[3];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_col_tiles = hex_round_up(ne1, 32) / 32;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_k_tiles   = hex_round_up(ne0, 32) / 32;
    ଅଚଳ size_t matrix_size = (size_t) n_col_tiles * n_k_tiles * tile_size;

    ଅଚଳ size_t row_size_bytes = ggml_row_size(t->type, ne0);
    ଅଚଳ size_t slice_size     = (size_t) ne1 * row_size_bytes;
    ଅଚଳ int64_t start_slice   = offset / slice_size;
    int64_t end_slice           = (offset + size + slice_size - 1) / slice_size;
    ଯଦି (end_slice > ne2 * ne3) {
        end_slice = ne2 * ne3;
    }

    ଅଚଳ int64_t row_stride = (ne0 / (32 / ELEMS_PER_KTILE));

    ପାଇଁ (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        ଅଚଳ size_t cur_start_byte     = (std::max)(offset, (size_t) slice_idx * slice_size);
        ଅଚଳ size_t cur_end_byte       = (std::min)(offset + size, (size_t) (slice_idx + 1) * slice_size);
        ଅଚଳ size_t slice_offset_start = cur_start_byte - (size_t) slice_idx * slice_size;
        ଅଚଳ size_t slice_offset_end   = cur_end_byte - (size_t) slice_idx * slice_size;

        ଅଚଳ int64_t start_row = slice_offset_start / row_size_bytes;
        ଅଚଳ int64_t end_row   = (std::min)((int64_t) ((slice_offset_end + row_size_bytes - 1) / row_size_bytes), ne1);

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା start_ct = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (start_row / 32);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା end_ct   = (std::min)((ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ((end_row + 31) / 32), n_col_tiles);

        DstElemT * dst_slice = (DstElemT *) (dst_bytes + (cur_start_byte - offset));
        ଅଚଳ uint8_t * matrix_src = (ଅଚଳ uint8_t *) t->data + slice_idx * matrix_size;

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ct = start_ct; ct < end_ct; ct++) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt = 0; kt < n_k_tiles; kt++) {
                ଅଚଳ uint8_t * tile_src = matrix_src + ((size_t) ct * n_k_tiles + kt) * tile_size;
                unpack_tile(dst_slice, tile_src, ne0, start_row, end_row, row_stride, ct, kt);
            }
        }
    }
}

// --- Block-32 Quantized Types (Q4_0, IQ4_NL, Q4_1, Q8_0, MXFP4) ---

ସ୍ଥିର ଶୂନ୍ୟ repack_q4_0_tiled(ggml_tensor * t, ଅଚଳ ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    repack_to_tiled_grid_32<block_q4_0, 1>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_Q4_0,
        [](uint8_t * tile_dst, ଅଚଳ block_q4_0 * src_slice, int64_t ne0, int64_t ne1, int64_t row_stride, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ct, ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
#ଯଦି defined(__ARM_NEON) || defined(__aarch64__)
            ଯଦି (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା row0 = 0; row0 < 32; row0 += 8) {
                    ଅଚଳ uint8_t * qs[8];
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_slice[(ct * 32 + row0 + rr) * row_stride + kt].qs;
                    }
                    repack_neon_4bit_rows(qs, tile_dst, row0);
                }

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = 0; row < 32; ++row) {
                    scale_dst[row] = src_slice[(ct * 32 + row) * row_stride + kt].d;
                }
                ଫେରନ୍ତୁ;
            }
#endif
            uint8_t tile_quants[32][32];
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = 0; row < 32; row++) {
                ଅଚଳ int64_t r = ct * 32 + row;
                ଯଦି (r < ne1 && kt < ne0 / 32) {
                    ଅଚଳ block_q4_0 & b = src_slice[r * row_stride + kt];
                    unpack_nibbles_32(tile_quants[row], b.qs);
                    scale_dst[row] = b.d;
                } ଅନ୍ୟଥା {
                    memset(tile_quants[row], 8, 32);
                    scale_dst[row] = 0;
                }
            }
            transpose_nibbles_to_tile(tile_dst, tile_quants);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ repack_tiled_q4_0(ଶୂନ୍ୟ * data, ଅଚଳ ggml_tensor * t, size_t offset, size_t size) {
    unpack_from_tiled_grid_32<block_q4_0, 1>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_Q4_0,
        [](block_q4_0 * dst_slice, ଅଚଳ uint8_t * tile_src, int64_t ne0, int64_t start_row, int64_t end_row, int64_t row_stride, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ct, ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_to_nibbles(tile_quants, tile_src);
            ଅଚଳ ggml_half * scale_src = (ଅଚଳ ggml_half *) (tile_src + 512);
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = 0; row < 32; row++) {
                ଅଚଳ int64_t r = ct * 32 + row;
                ଯଦି (r >= start_row && r < end_row && kt < ne0 / 32) {
                    block_q4_0 & b = dst_slice[(r - start_row) * row_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[row]);
                    b.d = scale_src[row];
                }
            }
        });
}

ସ୍ଥିର ଶୂନ୍ୟ repack_q4_1_tiled(ggml_tensor * t, ଅଚଳ ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    repack_to_tiled_grid_32<block_q4_1, 1>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_Q4_1,
        [](uint8_t * tile_dst, ଅଚଳ block_q4_1 * src_slice, int64_t ne0, int64_t ne1, int64_t row_stride, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ct, ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
#ଯଦି defined(__ARM_NEON) || defined(__aarch64__)
            ଯଦି (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା row0 = 0; row0 < 32; row0 += 8) {
                    ଅଚଳ uint8_t * qs[8];
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_slice[(ct * 32 + row0 + rr) * row_stride + kt].qs;
                    }
                    repack_neon_4bit_rows(qs, tile_dst, row0);
                }

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = 0; row < 32; ++row) {
                    ଅଚଳ block_q4_1 & src_block = src_slice[(ct * 32 + row) * row_stride + kt];
                    scale_dst[2 * row + 0] = src_block.d;
                    scale_dst[2 * row + 1] = src_block.m;
                }
                ଫେରନ୍ତୁ;
            }
#endif
            uint8_t tile_quants[32][32];
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = 0; row < 32; row++) {
                ଅଚଳ int64_t r = ct * 32 + row;
                ଯଦି (r < ne1 && kt < ne0 / 32) {
                    ଅଚଳ block_q4_1 & b = src_slice[r * row_stride + kt];
                    unpack_nibbles_32(tile_quants[row], b.qs);
                    scale_dst[2 * row + 0] = b.d;
                    scale_dst[2 * row + 1] = b.m;
                } ଅନ୍ୟଥା {
                    memset(tile_quants[row], 0, 32);
                    scale_dst[2 * row + 0] = 0;
                    scale_dst[2 * row + 1] = 0;
                }
            }
            transpose_nibbles_to_tile(tile_dst, tile_quants);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ repack_tiled_q4_1(ଶୂନ୍ୟ * data, ଅଚଳ ggml_tensor * t, size_t offset, size_t size) {
    unpack_from_tiled_grid_32<block_q4_1, 1>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_Q4_1,
        [](block_q4_1 * dst_slice, ଅଚଳ uint8_t * tile_src, int64_t ne0, int64_t start_row, int64_t end_row, int64_t row_stride, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ct, ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_to_nibbles(tile_quants, tile_src);
            ଅଚଳ ggml_half * scale_src = (ଅଚଳ ggml_half *) (tile_src + 512);
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = 0; row < 32; row++) {
                ଅଚଳ int64_t r = ct * 32 + row;
                ଯଦି (r >= start_row && r < end_row && kt < ne0 / 32) {
                    block_q4_1 & b = dst_slice[(r - start_row) * row_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[row]);
                    b.d = scale_src[2 * row + 0];
                    b.m = scale_src[2 * row + 1];
                }
            }
        });
}

ସ୍ଥିର ଶୂନ୍ୟ repack_mxfp4_tiled(ggml_tensor * t, ଅଚଳ ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    repack_to_tiled_grid_32<block_mxfp4, 1>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_MXFP4,
        [](uint8_t * tile_dst, ଅଚଳ block_mxfp4 * src_slice, int64_t ne0, int64_t ne1, int64_t row_stride, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ct, ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt) {
            uint8_t * scale_dst = tile_dst + 512;
#ଯଦି defined(__ARM_NEON) || defined(__aarch64__)
            ଯଦି (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା row0 = 0; row0 < 32; row0 += 8) {
                    ଅଚଳ uint8_t * qs[8];
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_slice[(ct * 32 + row0 + rr) * row_stride + kt].qs;
                    }
                    repack_neon_4bit_rows(qs, tile_dst, row0);
                }

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = 0; row < 32; ++row) {
                    scale_dst[row] = src_slice[(ct * 32 + row) * row_stride + kt].e;
                }
                ଫେରନ୍ତୁ;
            }
#endif
            uint8_t tile_quants[32][32];
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = 0; row < 32; row++) {
                ଅଚଳ int64_t r = ct * 32 + row;
                ଯଦି (r < ne1 && kt < ne0 / 32) {
                    ଅଚଳ block_mxfp4 & b = src_slice[r * row_stride + kt];
                    unpack_nibbles_32(tile_quants[row], b.qs);
                    scale_dst[row] = b.e;
                } ଅନ୍ୟଥା {
                    memset(tile_quants[row], 0, 32);
                    scale_dst[row] = 0;
                }
            }
            transpose_nibbles_to_tile(tile_dst, tile_quants);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ repack_tiled_mxfp4(ଶୂନ୍ୟ * data, ଅଚଳ ggml_tensor * t, size_t offset, size_t size) {
    unpack_from_tiled_grid_32<block_mxfp4, 1>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_MXFP4,
        [](block_mxfp4 * dst_slice, ଅଚଳ uint8_t * tile_src, int64_t ne0, int64_t start_row, int64_t end_row, int64_t row_stride, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ct, ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_to_nibbles(tile_quants, tile_src);
            ଅଚଳ uint8_t * scale_src = tile_src + 512;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = 0; row < 32; row++) {
                ଅଚଳ int64_t r = ct * 32 + row;
                ଯଦି (r >= start_row && r < end_row && kt < ne0 / 32) {
                    block_mxfp4 & b = dst_slice[(r - start_row) * row_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[row]);
                    b.e = scale_src[row];
                }
            }
        });
}

ସ୍ଥିର ଶୂନ୍ୟ repack_q8_0_tiled(ggml_tensor * t, ଅଚଳ ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    repack_to_tiled_grid_32<block_q8_0, 1>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_Q8_0,
        [](uint8_t * tile_dst, ଅଚଳ block_q8_0 * src_slice, int64_t ne0, int64_t ne1, int64_t row_stride, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ct, ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 1024);
#ଯଦି defined(__ARM_NEON) || defined(__aarch64__)
            ଯଦି (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା row0 = 0; row0 < 32; row0 += 8) {
                    ଅଚଳ uint8_t * qs[8];
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା rr = 0; rr < 8; ++rr) {
                        qs[rr] = (ଅଚଳ uint8_t *) src_slice[(ct * 32 + row0 + rr) * row_stride + kt].qs;
                    }
                    repack_neon_q8_rows(qs, tile_dst, row0);
                }

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = 0; row < 32; ++row) {
                    scale_dst[row] = src_slice[(ct * 32 + row) * row_stride + kt].d;
                }
                ଫେରନ୍ତୁ;
            }
#endif
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା cp = 0; cp < 16; cp++) {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା col0 = cp * 2;
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା col1 = col0 + 1;
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = 0; row < 32; row++) {
                    ଅଚଳ int64_t r = ct * 32 + row;
                    ଅଚଳ block_q8_0 * b = (r < ne1 && kt < ne0 / 32) ? &src_slice[r * row_stride + kt] : nullptr;
                    tile_dst[cp * 64 + 2 * row + 0] = b ? b->qs[col0] : 0;
                    tile_dst[cp * 64 + 2 * row + 1] = b ? b->qs[col1] : 0;
                }
            }
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = 0; row < 32; row++) {
                ଅଚଳ int64_t r = ct * 32 + row;
                scale_dst[row] = (r < ne1 && kt < ne0 / 32) ? src_slice[r * row_stride + kt].d : 0;
            }
        });
}

ସ୍ଥିର ଶୂନ୍ୟ repack_tiled_q8_0(ଶୂନ୍ୟ * data, ଅଚଳ ggml_tensor * t, size_t offset, size_t size) {
    unpack_from_tiled_grid_32<block_q8_0, 1>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_Q8_0,
        [](block_q8_0 * dst_slice, ଅଚଳ uint8_t * tile_src, int64_t ne0, int64_t start_row, int64_t end_row, int64_t row_stride, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ct, ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା cp = 0; cp < 16; cp++) {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା col0 = cp * 2;
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା col1 = col0 + 1;
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = 0; row < 32; row++) {
                    ଅଚଳ int64_t r = ct * 32 + row;
                    ଯଦି (r >= start_row && r < end_row && kt < ne0 / 32) {
                        block_q8_0 & b = dst_slice[(r - start_row) * row_stride + kt];
                        b.qs[col0] = tile_src[cp * 64 + 2 * row + 0];
                        b.qs[col1] = tile_src[cp * 64 + 2 * row + 1];
                    }
                }
            }
            ଅଚଳ ggml_half * scale_src = (ଅଚଳ ggml_half *) (tile_src + 1024);
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = 0; row < 32; row++) {
                ଅଚଳ int64_t r = ct * 32 + row;
                ଯଦି (r >= start_row && r < end_row && kt < ne0 / 32) {
                    dst_slice[(r - start_row) * row_stride + kt].d = scale_src[row];
                }
            }
        });
}

// --- Float Types (F16, F32, BF16 -> f16_tiled HMX layout) ---

ସ୍ଥିର ଶୂନ୍ୟ repack_q2_0_tiled(ggml_tensor * t, ଅଚଳ ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(t->ne[0] % 64 == 0);
    GGML_ASSERT(size == ggml_nbytes(t));

    ଅଚଳ block_q2_0 * src = (ଅଚଳ block_q2_0 *) data;
    ଅଚଳ int64_t blocks_per_row = t->ne[0] / 64;
    ଅଚଳ int64_t n_col_tiles = hex_round_up(t->ne[1], 32) / 32;
    ଅଚଳ int64_t n_weight_tiles = blocks_per_row;
    ଅଚଳ size_t matrix_size = (size_t) n_col_tiles * n_weight_tiles * HTP_TENSOR_TILE_SIZE_Q2_0;

    ପାଇଁ (int64_t slice = 0; slice < t->ne[2] * t->ne[3]; ++slice) {
        ଅଚଳ block_q2_0 * src_slice = src + slice * t->ne[1] * blocks_per_row;
        uint8_t * dst_slice = (uint8_t *) t->data + slice * matrix_size;
        ଅଚଳ size_t ct_bytes = (size_t) n_weight_tiles * HTP_TENSOR_TILE_SIZE_Q2_0;

        repack_parallel_for_ct(matrix_size, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) n_col_tiles, [&](ପୂର୍ଣ୍ଣସଂଖ୍ୟା ct0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ct1) {
            memset(dst_slice + (size_t) ct0 * ct_bytes, 0, (size_t) (ct1 - ct0) * ct_bytes);
            ଅଚଳ int64_t r0 = (int64_t) ct0 * 32;
            ଅଚଳ int64_t r1 = (std::min)((int64_t) ct1 * 32, t->ne[1]);
            ପାଇଁ (int64_t row_index = r0; row_index < r1; ++row_index) {
                ଅଚଳ int64_t ct = row_index / 32;
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (row_index % 32);
                ପାଇଁ (int64_t kt = 0; kt < n_weight_tiles; ++kt) {
                    ଅଚଳ block_q2_0 * block = src_slice + row_index * blocks_per_row + kt;
                    uint8_t * tile = dst_slice + (ct * n_weight_tiles + kt) * HTP_TENSOR_TILE_SIZE_Q2_0;
#ଯଦି defined(__ARM_NEON) || defined(__aarch64__)
                    repack_neon_q2_0_block(block->qs, tile, row);
#ଅନ୍ୟଥା
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା group = 0; group < 16; ++group) {
                        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା byte = 0; byte < 4; ++byte) {
                            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 4 * group + byte;
                            ଅଚଳ uint8_t code = (block->qs[k / 4] >> (2 * (k % 4))) & 3;
                            tile[(group >> 2) * 128 + 4 * row + byte] |= code << (2 * (group & 3));
                        }
                    }
#endif
                    memcpy(tile + 512 + 2 * row, &block->d, ଆକାର(block->d));
                }
            }
        });
    }
}

ସ୍ଥିର ଶୂନ୍ୟ repack_tiled_q2_0(ଶୂନ୍ୟ * data, ଅଚଳ ggml_tensor * t, size_t offset, size_t size) {
    GGML_ASSERT(t->ne[0] % 64 == 0);
    GGML_ASSERT(offset + size <= ggml_nbytes(t));

    ଅଚଳ size_t first_block = offset / ଆକାର(block_q2_0);
    ଅଚଳ size_t end_block = (offset + size + ଆକାର(block_q2_0) - 1) / ଆକାର(block_q2_0);
    ଅଚଳ size_t blocks_per_row = t->ne[0] / 64;
    ଅଚଳ size_t blocks_per_slice = t->ne[1] * blocks_per_row;
    ଅଚଳ size_t n_weight_tiles = blocks_per_row;
    ଅଚଳ size_t n_col_tiles = hex_round_up(t->ne[1], 32) / 32;
    ଅଚଳ size_t matrix_size = n_col_tiles * n_weight_tiles * HTP_TENSOR_TILE_SIZE_Q2_0;

    ପାଇଁ (size_t block_index = first_block; block_index < end_block; ++block_index) {
        ଅଚଳ size_t slice = block_index / blocks_per_slice;
        ଅଚଳ size_t in_slice = block_index % blocks_per_slice;
        ଅଚଳ size_t row_index = in_slice / blocks_per_row;
        ଅଚଳ size_t kt = in_slice % blocks_per_row;
        ଅଚଳ size_t ct = row_index / 32;
        ଅଚଳ size_t row = row_index % 32;
        ଅଚଳ uint8_t * tile = (ଅଚଳ uint8_t *) t->data + slice * matrix_size +
            (ct * n_weight_tiles + kt) * HTP_TENSOR_TILE_SIZE_Q2_0;

        block_q2_0 block = {};
        memcpy(&block.d, tile + 512 + 2 * row, ଆକାର(block.d));
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା group = 0; group < 16; ++group) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା byte = 0; byte < 4; ++byte) {
                ଅଚଳ uint8_t packed = tile[(group >> 2) * 128 + 4 * row + byte];
                ଅଚଳ uint8_t code = (packed >> (2 * (group & 3))) & 3;
                block.qs[group] |= code << (2 * byte);
            }
        }

        ଅଚଳ size_t block_begin = block_index * ଆକାର(block_q2_0);
        ଅଚଳ size_t copy_begin = (std::max)(block_begin, offset);
        ଅଚଳ size_t copy_end = (std::min)(block_begin + ଆକାର(block_q2_0), offset + size);
        memcpy((uint8_t *) data + copy_begin - offset, (ଅଚଳ uint8_t *) &block + copy_begin - block_begin, copy_end - copy_begin);
    }
}

template <typename SrcT, typename ToFp16Fn>
ସ୍ଥିର ଶୂନ୍ୟ repack_float_to_f16_tiled(ggml_tensor * t, ଅଚଳ ଶୂନ୍ୟ * data, size_t offset, size_t size, ToFp16Fn && to_fp16) {
    repack_to_tiled_grid_32<SrcT, 32>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_F16,
        [&](uint8_t * tile_bytes, ଅଚଳ SrcT * src_slice, int64_t ne0, int64_t ne1, int64_t row_stride, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ct, ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt) {
            uint16_t * tile_dst = (uint16_t *) tile_bytes;
#ଯଦି defined(__ARM_NEON) || defined(__aarch64__)
            ଯଦି (ct * 32 + 32 <= ne1 && kt * 32 + 32 <= ne0) {
                repack_neon_f16_tile(tile_dst, src_slice, row_stride, ct, kt);
                ଫେରନ୍ତୁ;
            }
#endif
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା cp = 0; cp < 16; cp++) {
                ଅଚଳ int64_t c0 = kt * 32 + cp * 2;
                ଅଚଳ int64_t c1 = c0 + 1;
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = 0; row < 32; row++) {
                    ଅଚଳ int64_t r = ct * 32 + row;
                    ଅଚଳ SrcT * r_ptr = (r < ne1) ? &src_slice[r * row_stride] : nullptr;
                    tile_dst[cp * 64 + row * 2 + 0] = (r_ptr && c0 < ne0) ? to_fp16(r_ptr[c0]) : 0;
                    tile_dst[cp * 64 + row * 2 + 1] = (r_ptr && c1 < ne0) ? to_fp16(r_ptr[c1]) : 0;
                }
            }
        });
}

template <typename DstT, typename FromFp16Fn>
ସ୍ଥିର ଶୂନ୍ୟ unpack_f16_tiled_to_float(ଶୂନ୍ୟ * data, ଅଚଳ ggml_tensor * t, size_t offset, size_t size, FromFp16Fn && from_fp16) {
    unpack_from_tiled_grid_32<DstT, 32>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_F16,
        [&](DstT * dst_slice, ଅଚଳ uint8_t * tile_bytes, int64_t ne0, int64_t start_row, int64_t end_row, int64_t row_stride, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ct, ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt) {
            ଅଚଳ uint16_t * tile_src = (ଅଚଳ uint16_t *) tile_bytes;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା cp = 0; cp < 16; cp++) {
                ଅଚଳ int64_t c0 = kt * 32 + cp * 2;
                ଅଚଳ int64_t c1 = c0 + 1;
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = 0; row < 32; row++) {
                    ଅଚଳ int64_t r = ct * 32 + row;
                    ଯଦି (r >= start_row && r < end_row) {
                        DstT * r_ptr = dst_slice + (r - start_row) * row_stride;
                        ଯଦି (c0 < ne0) r_ptr[c0] = from_fp16(tile_src[cp * 64 + row * 2 + 0]);
                        ଯଦି (c1 < ne0) r_ptr[c1] = from_fp16(tile_src[cp * 64 + row * 2 + 1]);
                    }
                }
            }
        });
}

// --- Generic K-Quant (super-block 256) grid walkers ---

template <typename BlockT, typename PackKtFn>
ସ୍ଥିର ଶୂନ୍ୟ repack_kquant_to_tiled_grid(ggml_tensor * t, ଅଚଳ ଶୂନ୍ୟ * data, size_t offset, size_t size,
                                        size_t tile_size, PackKtFn && pack_kt) {
    GGML_ASSERT(offset == 0);
    GGML_UNUSED(size);

    ଅଚଳ BlockT * src_matrix = (ଅଚଳ BlockT *) data;
    ଅଚଳ int64_t ne0 = t->ne[0];
    ଅଚଳ int64_t ne1 = t->ne[1];
    ଅଚଳ int64_t ne2 = t->ne[2];
    ଅଚଳ int64_t ne3 = t->ne[3];
    GGML_ASSERT(ne0 % QK_K == 0);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_col_tiles    = hex_round_up(ne1, 32) / 32;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_k_tiles      = hex_round_up(ne0, 32) / 32;
    ଅଚଳ size_t matrix_size = (size_t) n_col_tiles * n_k_tiles * tile_size;
    ଅଚଳ int64_t sb_per_row = ne0 / QK_K;

    ପାଇଁ (int64_t slice_idx = 0; slice_idx < ne2 * ne3; slice_idx++) {
        ଅଚଳ BlockT * src_slice = src_matrix + slice_idx * (ne1 * sb_per_row);
        uint8_t * matrix_dst     = (uint8_t *) t->data + slice_idx * matrix_size;
        ଅଚଳ size_t ct_bytes    = (size_t) n_k_tiles * tile_size;

        repack_parallel_for_ct(matrix_size, n_col_tiles, [&](ପୂର୍ଣ୍ଣସଂଖ୍ୟା ct0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ct1) {
            // Padding rows and OR-ed bit planes require zeroed tiles
            memset(matrix_dst + (size_t) ct0 * ct_bytes, 0, (size_t) (ct1 - ct0) * ct_bytes);

            ଅଚଳ int64_t r0 = (int64_t) ct0 * 32;
            ଅଚଳ int64_t r1 = (std::min)((int64_t) ct1 * 32, ne1);
            ପାଇଁ (int64_t r = r0; r < r1; r++) {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ct  = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (r / 32);
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (r % 32);
                ଅଚଳ BlockT * src_row = src_slice + r * sb_per_row;

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile = matrix_dst + ((size_t) ct * n_k_tiles + kt) * tile_size;
                    pack_kt(tile, &src_row[kt / 8], row, kt % 8);
                }
            }
        });
    }
}

template <typename BlockT, typename UnpackSbFn>
ସ୍ଥିର ଶୂନ୍ୟ unpack_kquant_from_tiled_grid(ଶୂନ୍ୟ * data, ଅଚଳ ggml_tensor * t, size_t offset, size_t size,
                                          size_t tile_size, UnpackSbFn && unpack_sb) {
    uint8_t * dst_bytes = (uint8_t *) data;
    ଅଚଳ int64_t ne0 = t->ne[0];
    ଅଚଳ int64_t ne1 = t->ne[1];
    ଅଚଳ int64_t ne2 = t->ne[2];
    ଅଚଳ int64_t ne3 = t->ne[3];
    GGML_ASSERT(ne0 % QK_K == 0);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_col_tiles    = hex_round_up(ne1, 32) / 32;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_k_tiles      = hex_round_up(ne0, 32) / 32;
    ଅଚଳ size_t matrix_size = (size_t) n_col_tiles * n_k_tiles * tile_size;
    ଅଚଳ int64_t sb_per_row = ne0 / QK_K;

    ଅଚଳ size_t row_size_bytes = ggml_row_size(t->type, ne0);
    ଅଚଳ size_t slice_size     = (size_t) ne1 * row_size_bytes;
    ଅଚଳ int64_t start_slice   = offset / slice_size;
    int64_t end_slice           = (offset + size + slice_size - 1) / slice_size;
    ଯଦି (end_slice > ne2 * ne3) {
        end_slice = ne2 * ne3;
    }

    ପାଇଁ (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        ଅଚଳ size_t cur_start_byte     = (std::max)(offset, (size_t) slice_idx * slice_size);
        ଅଚଳ size_t cur_end_byte       = (std::min)(offset + size, (size_t) (slice_idx + 1) * slice_size);
        ଅଚଳ size_t slice_offset_start = cur_start_byte - (size_t) slice_idx * slice_size;
        ଅଚଳ size_t slice_offset_end   = cur_end_byte - (size_t) slice_idx * slice_size;

        ଅଚଳ int64_t start_row = slice_offset_start / row_size_bytes;
        ଅଚଳ int64_t end_row   = (std::min)((int64_t) ((slice_offset_end + row_size_bytes - 1) / row_size_bytes), ne1);

        BlockT * dst_slice         = (BlockT *) (dst_bytes + (cur_start_byte - offset));
        ଅଚଳ uint8_t * matrix_src = (ଅଚଳ uint8_t *) t->data + slice_idx * matrix_size;

        ପାଇଁ (int64_t r = start_row; r < end_row; r++) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ct  = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (r / 32);
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (r % 32);
            BlockT * dst_row = dst_slice + (r - start_row) * sb_per_row;

            ପାଇଁ (int64_t sb = 0; sb < sb_per_row; sb++) {
                BlockT * b = &dst_row[sb];
                memset(b, 0, ଆକାର(BlockT));
                ଅଚଳ uint8_t * sb_tiles = matrix_src + ((size_t) ct * n_k_tiles + sb * 8) * tile_size;
                unpack_sb(b, sb_tiles, tile_size, row);
            }
        }
    }
}

// --- K-Quant Helpers & Per-Type Implementations (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) ---

#ଯଦି !defined(__ARM_NEON) && !defined(__aarch64__)
// unsigned 6-bit value (0..63) of element e of a Q6_K block, same bit layout as dequantize_row_q6_K
ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint8_t q6_K_get_quant(ଅଚଳ block_q6_K * b, ପୂର୍ଣ୍ଣସଂଖ୍ୟା e) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା c = e / 128;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା w = e % 128;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g = w / 32;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା l = w % 32;
    ଅଚଳ uint8_t * ql = b->ql + c * 64;
    ଅଚଳ uint8_t * qh = b->qh + c * 32;
    uint8_t lo, hi;
    ବାଛ (g) {
        ସ୍ଥିତି 0:  lo = ql[l]      & 0xF; hi = (qh[l] >> 0) & 3; ଭାଙ୍ଗ;
        ସ୍ଥିତି 1:  lo = ql[l + 32] & 0xF; hi = (qh[l] >> 2) & 3; ଭାଙ୍ଗ;
        ସ୍ଥିତି 2:  lo = ql[l]      >> 4;  hi = (qh[l] >> 4) & 3; ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: lo = ql[l + 32] >> 4;  hi = (qh[l] >> 6) & 3; ଭାଙ୍ଗ;
    }
    ଫେରନ୍ତୁ (uint8_t) (lo | (hi << 4));
}
#endif

ସ୍ଥିର ଶୂନ୍ୟ repack_q6_K_tiled(ggml_tensor * t, ଅଚଳ ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    repack_kquant_to_tiled_grid<block_q6_K>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_Q6_K,
        [](uint8_t * tile, ଅଚଳ block_q6_K * b, ପୂର୍ଣ୍ଣସଂଖ୍ୟା row, ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt_local) {
            ଅଚଳ ଦଶମିକ d     = GGML_FP16_TO_FP32(b->d);
            uint8_t * lo_pl   = tile;
            uint8_t * hi_pl   = tile + 512;
            ggml_half * sc_pl = (ggml_half *) (tile + 768);

#ଯଦି defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q6_k(b, kt_local, lo_pl, hi_pl, row);
#ଅନ୍ୟଥା
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା lk = 0; lk < 32; lk++) {
                ଅଚଳ uint8_t q6 = q6_K_get_quant(b, kt_local * 32 + lk);
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g   = lk >> 2;
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା pos = row * 4 + (lk & 3);
                lo_pl[(g >> 1) * 128 + pos] |= (uint8_t) ((q6 & 0xF) << ((g & 1) * 4));
                hi_pl[(g >> 2) * 128 + pos] |= (uint8_t) ((q6 >> 4) << ((g & 3) * 2));
            }
#endif
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + row] = GGML_FP32_TO_FP16(d * (ଦଶମିକ) b->scales[kt_local * 2 + sub]);
            }
        });
}

ସ୍ଥିର ଶୂନ୍ୟ repack_tiled_q6_K(ଶୂନ୍ୟ * data, ଅଚଳ ggml_tensor * t, size_t offset, size_t size) {
    unpack_kquant_from_tiled_grid<block_q6_K>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_Q6_K,
        [](block_q6_K * b, ଅଚଳ uint8_t * sb_tiles, size_t tile_size, ପୂର୍ଣ୍ଣସଂଖ୍ୟା row) {
            ଦଶମିକ sub_scales[16];
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt_local = 0; kt_local < 8; kt_local++) {
                ଅଚଳ uint8_t *   tile  = sb_tiles + kt_local * tile_size;
                ଅଚଳ uint8_t *   lo_pl = tile;
                ଅଚଳ uint8_t *   hi_pl = tile + 512;
                ଅଚଳ ggml_half * sc_pl = (ଅଚଳ ggml_half *) (tile + 768);

                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା c  = kt_local / 4;
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g  = kt_local % 4;
                uint8_t * ql = b->ql + c * 64;
                uint8_t * qh = b->qh + c * 32;

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା lk = 0; lk < 32; lk++) {
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_tile = lk >> 2;
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା pos    = row * 4 + (lk & 3);
                    ଅଚଳ uint8_t lo = (lo_pl[(g_tile >> 1) * 128 + pos] >> ((g_tile & 1) * 4)) & 0xF;
                    ଅଚଳ uint8_t hi = (hi_pl[(g_tile >> 2) * 128 + pos] >> ((g_tile & 3) * 2)) & 3;

                    ବାଛ (g) {
                        ସ୍ଥିତି 0:  ql[lk]      |= lo;        qh[lk] |= (hi << 0); ଭାଙ୍ଗ;
                        ସ୍ଥିତି 1:  ql[lk + 32] |= lo;        qh[lk] |= (hi << 2); ଭାଙ୍ଗ;
                        ସ୍ଥିତି 2:  ql[lk]      |= (lo << 4); qh[lk] |= (hi << 4); ଭାଙ୍ଗ;
                        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: ql[lk + 32] |= (lo << 4); qh[lk] |= (hi << 6); ଭାଙ୍ଗ;
                    }
                }

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା sub = 0; sub < 2; sub++) {
                    sub_scales[kt_local * 2 + sub] = GGML_FP16_TO_FP32(sc_pl[sub * 32 + row]);
                }
            }

            ଦଶମିକ max_abs_scale = 0.0f;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = 0; s < 16; s++) {
                max_abs_scale = (std::max)(max_abs_scale, fabsf(sub_scales[s]));
            }

            ଯଦି (max_abs_scale == 0.0f) {
                b->d = GGML_FP32_TO_FP16(0.0f);
            } ଅନ୍ୟଥା {
                b->d = GGML_FP32_TO_FP16(max_abs_scale / 127.0f);
                ଅଚଳ ଦଶମିକ d_actual = GGML_FP16_TO_FP32(b->d);
                ଅଚଳ ଦଶମିକ inv_d    = (d_actual != 0.0f) ? (1.0f / d_actual) : 0.0f;
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = 0; s < 16; s++) {
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା sc = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) roundf(sub_scales[s] * inv_d);
                    b->scales[s] = (int8_t) (std::max)(-128, (std::min)(127, sc));
                }
            }
        });
}

#ଯଦି !defined(__ARM_NEON) && !defined(__aarch64__)
// low 2 bits (0..3) of element e of a Q2_K or Q3_K block, same bit layout as dequantize_row_q2_K / q3_K
ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint8_t q2_3_K_get_low2(ଅଚଳ uint8_t * qs, ପୂର୍ଣ୍ଣସଂଖ୍ୟା e) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା c = e / 128;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = (e % 128) / 32;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା l = e % 32;
    ଫେରନ୍ତୁ (qs[c * 32 + l] >> (2 * j)) & 3;
}

// hmask bit of element e of a Q3_K block, same bit layout as dequantize_row_q3_K
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ତାର୍କିକ q3_K_get_hbit(ଅଚଳ block_q3_K * b, ପୂର୍ଣ୍ଣସଂଖ୍ୟା e) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା c = e / 128;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = (e % 128) / 32;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା l = e % 32;
    ଫେରନ୍ତୁ (b->hmask[l] >> (c * 4 + j)) & 1;
}
#endif

// signed 6-bit scale j (-32..31) of a Q3_K block, same packing as quantize_row_q3_K_ref
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା q3_K_get_scale(ଅଚଳ uint8_t * scales, ପୂର୍ଣ୍ଣସଂଖ୍ୟା j) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା lo = (j < 8) ? (scales[j] & 0xF) : (scales[j - 8] >> 4);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା hi = (scales[8 + j % 4] >> (2 * (j / 4))) & 3;
    ଫେରନ୍ତୁ (lo | (hi << 4)) - 32;
}

// read-back: find fp16 d and l[j] in [lmin, lmax] with fp16(d * l[j]) == prod[j] for all j, false if none
ସ୍ଥିର ତାର୍କିକ hexagon_recover_k_scales(ଅଚଳ ggml_half * prod, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ପୂର୍ଣ୍ଣସଂଖ୍ୟା lmin, ପୂର୍ଣ୍ଣସଂଖ୍ୟା lmax, ggml_half * d_out, ପୂର୍ଣ୍ଣସଂଖ୍ୟା * l_out) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା jmax = 0;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 1; j < n; j++) {
        ଯଦି (fabsf(GGML_FP16_TO_FP32(prod[j])) > fabsf(GGML_FP16_TO_FP32(prod[jmax]))) {
            jmax = j;
        }
    }
    ଅଚଳ ଦଶମିକ pmax = GGML_FP16_TO_FP32(prod[jmax]);
    ଯଦି (pmax == 0.0f) {
        *d_out = GGML_FP32_TO_FP16(0.0f);
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < n; j++) {
            l_out[j] = 0;
        }
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    // the quantizers put the largest scale at or near the range end, so try large |l| first
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା lext = (std::max)(-lmin, lmax);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା a = lext; a >= 1; a--) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା sign : { 1, -1 }) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା lj = sign * a;
            ଯଦି (lj < lmin || lj > lmax) {
                ଜାରି;
            }
            ଅଚଳ ggml_half d0 = GGML_FP32_TO_FP16(pmax / (ଦଶମିକ) lj);
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ulp : { 0, -1, 1 }) {
                ggml_half d = d0;
                uint16_t bits;
                memcpy(&bits, &d, ଆକାର(bits));
                bits = (uint16_t) (bits + ulp);
                memcpy(&d, &bits, ଆକାର(bits));

                ଅଚଳ ଦଶମିକ df = GGML_FP16_TO_FP32(d);
                ଯଦି (!std::isfinite(df) || df == 0.0f) {
                    ଜାରି;
                }
                ତାର୍କିକ ok = ସତ୍ୟ;
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < n && ok; j++) {
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା l = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) roundf(GGML_FP16_TO_FP32(prod[j]) / df);
                    ଅଚଳ ggml_half p = GGML_FP32_TO_FP16(df * (ଦଶମିକ) l);
                    ok = l >= lmin && l <= lmax && memcmp(&p, &prod[j], ଆକାର(p)) == 0;
                    l_out[j] = l;
                }
                ଯଦି (ok) {
                    *d_out = d;
                    ଫେରନ୍ତୁ ସତ୍ୟ;
                }
            }
        }
    }
    ଫେରନ୍ତୁ ମିଥ୍ୟା;
}

ସ୍ଥିର ଶୂନ୍ୟ repack_q3_K_tiled(ggml_tensor * t, ଅଚଳ ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    repack_kquant_to_tiled_grid<block_q3_K>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_Q3_K,
        [](uint8_t * tile, ଅଚଳ block_q3_K * b, ପୂର୍ଣ୍ଣସଂଖ୍ୟା row, ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt_local) {
            ଅଚଳ ଦଶମିକ d     = GGML_FP16_TO_FP32(b->d);
            uint8_t * lo_pl   = tile;
            uint8_t * neg_pl  = tile + 256;
            ggml_half * sc_pl = (ggml_half *) (tile + 384);

#ଯଦି defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q2_3_k_low2(b->qs, kt_local, lo_pl, row);
            repack_neon_q3_k_hmask(b->hmask, kt_local, neg_pl, row);
#ଅନ୍ୟଥା
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା lk = 0; lk < 32; lk++) {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା e   = kt_local * 32 + lk;
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g   = lk >> 2;
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା pos = row * 4 + (lk & 3);
                lo_pl[(g >> 2) * 128 + pos] |= (uint8_t) (q2_3_K_get_low2(b->qs, e) << ((g & 3) * 2));
                ଯଦି (!q3_K_get_hbit(b, e)) {
                    neg_pl[pos] |= (uint8_t) (1 << g);
                }
            }
#endif
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + row] = GGML_FP32_TO_FP16(d * (ଦଶମିକ) q3_K_get_scale(b->scales, kt_local * 2 + sub));
            }
        });
}

ସ୍ଥିର ଶୂନ୍ୟ repack_tiled_q3_K(ଶୂନ୍ୟ * data, ଅଚଳ ggml_tensor * t, size_t offset, size_t size) {
    unpack_kquant_from_tiled_grid<block_q3_K>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_Q3_K,
        [](block_q3_K * b, ଅଚଳ uint8_t * sb_tiles, size_t tile_size, ପୂର୍ଣ୍ଣସଂଖ୍ୟା row) {
            ggml_half sub_scales[16];
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt_local = 0; kt_local < 8; kt_local++) {
                ଅଚଳ uint8_t *   tile   = sb_tiles + kt_local * tile_size;
                ଅଚଳ uint8_t *   lo_pl  = tile;
                ଅଚଳ uint8_t *   neg_pl = tile + 256;
                ଅଚଳ ggml_half * sc_pl  = (ଅଚଳ ggml_half *) (tile + 384);

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା lk = 0; lk < 32; lk++) {
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା e   = kt_local * 32 + lk;
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g   = lk >> 2;
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା pos = row * 4 + (lk & 3);
                    ଅଚଳ uint8_t lo = (lo_pl[(g >> 2) * 128 + pos] >> ((g & 3) * 2)) & 3;

                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା c = e / 128;
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = (e % 128) / 32;
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା l = e % 32;
                    b->qs[c * 32 + l] |= (uint8_t) (lo << (2 * j));
                    ଯଦି (!((neg_pl[pos] >> g) & 1)) {
                        b->hmask[l] |= (uint8_t) (1 << (c * 4 + j));
                    }
                }

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା sub = 0; sub < 2; sub++) {
                    sub_scales[kt_local * 2 + sub] = sc_pl[sub * 32 + row];
                }
            }

            ପୂର୍ଣ୍ଣସଂଖ୍ୟା ls[16];
            ଯଦି (!hexagon_recover_k_scales(sub_scales, 16, -32, 31, &b->d, ls)) {
                ଦଶମିକ max_scale = 0.0f;
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = 0; s < 16; s++) {
                    ଯଦି (fabsf(GGML_FP16_TO_FP32(sub_scales[s])) > fabsf(max_scale)) {
                        max_scale = GGML_FP16_TO_FP32(sub_scales[s]);
                    }
                }
                b->d = GGML_FP32_TO_FP16(-max_scale / 32.0f);
                ଅଚଳ ଦଶମିକ d_actual = GGML_FP16_TO_FP32(b->d);
                ଅଚଳ ଦଶମିକ inv_d    = (d_actual != 0.0f) ? (1.0f / d_actual) : 0.0f;
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = 0; s < 16; s++) {
                    ls[s] = (std::max)(-32, (std::min)(31, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) roundf(GGML_FP16_TO_FP32(sub_scales[s]) * inv_d)));
                }
            }

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = 0; s < 16; s++) {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା l = ls[s] + 32;
                ଯଦି (s < 8) {
                    b->scales[s] = l & 0xF;
                } ଅନ୍ୟଥା {
                    b->scales[s - 8] |= (uint8_t) ((l & 0xF) << 4);
                }
                b->scales[s % 4 + 8] |= (uint8_t) ((l >> 4) << (2 * (s / 4)));
            }
        });
}

ସ୍ଥିର ଶୂନ୍ୟ repack_q2_K_tiled(ggml_tensor * t, ଅଚଳ ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    repack_kquant_to_tiled_grid<block_q2_K>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_Q2_K,
        [](uint8_t * tile, ଅଚଳ block_q2_K * b, ପୂର୍ଣ୍ଣସଂଖ୍ୟା row, ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt_local) {
            uint8_t * lo_pl   = tile;
            uint8_t * sc_pl   = tile + 256;
            ggml_half * dd_pl = (ggml_half *) (tile + 320);

#ଯଦି defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q2_3_k_low2(b->qs, kt_local, lo_pl, row);
#ଅନ୍ୟଥା
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା lk = 0; lk < 32; lk++) {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g   = lk >> 2;
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା pos = row * 4 + (lk & 3);
                lo_pl[(g >> 2) * 128 + pos] |= (uint8_t) (q2_3_K_get_low2(b->qs, kt_local * 32 + lk) << ((g & 3) * 2));
            }
#endif
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + row] = b->scales[kt_local * 2 + sub];
            }
            ଯଦି (row / 8 == kt_local % 4) {
                dd_pl[row % 8] = kt_local < 4 ? b->d : (ggml_half) (b->dmin ^ 0x8000);
            }
        });
}

ସ୍ଥିର ଶୂନ୍ୟ repack_tiled_q2_K(ଶୂନ୍ୟ * data, ଅଚଳ ggml_tensor * t, size_t offset, size_t size) {
    unpack_kquant_from_tiled_grid<block_q2_K>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_Q2_K,
        [](block_q2_K * b, ଅଚଳ uint8_t * sb_tiles, size_t tile_size, ପୂର୍ଣ୍ଣସଂଖ୍ୟା row) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt_local = 0; kt_local < 8; kt_local++) {
                ଅଚଳ uint8_t *   tile  = sb_tiles + kt_local * tile_size;
                ଅଚଳ uint8_t *   lo_pl = tile;
                ଅଚଳ uint8_t *   sc_pl = tile + 256;
                ଅଚଳ ggml_half * dd_pl = (ଅଚଳ ggml_half *) (tile + 320);

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା lk = 0; lk < 32; lk++) {
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା e   = kt_local * 32 + lk;
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g   = lk >> 2;
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା pos = row * 4 + (lk & 3);
                    ଅଚଳ uint8_t lo = (lo_pl[(g >> 2) * 128 + pos] >> ((g & 3) * 2)) & 3;
                    b->qs[(e / 128) * 32 + e % 32] |= (uint8_t) (lo << (2 * ((e % 128) / 32)));
                }

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା sub = 0; sub < 2; sub++) {
                    b->scales[kt_local * 2 + sub] = sc_pl[sub * 32 + row];
                }
                ଯଦି (row / 8 == kt_local % 4) {
                    ଯଦି (kt_local < 4) {
                        b->d = dd_pl[row % 8];
                    } ଅନ୍ୟଥା {
                        b->dmin = (ggml_half) (dd_pl[row % 8] ^ 0x8000);
                    }
                }
            }
        });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ get_scale_min_k4(ପୂର୍ଣ୍ଣସଂଖ୍ୟା j, ଅଚଳ uint8_t * q, uint8_t * d, uint8_t * m) {
    ଯଦି (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } ଅନ୍ୟଥା {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ pack_k4_scales_mins(uint8_t * scales, ggml_half * d, ggml_half * dmin,
                                       ଅଚଳ ଦଶମିକ * sub_scales, ଅଚଳ ଦଶମିକ * sub_mins) {
    ଦଶମିକ max_scale = 0.0f;
    ଦଶମିକ max_min   = 0.0f;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < 8; j++) {
        ଯଦି (sub_scales[j] > max_scale) max_scale = sub_scales[j];
        ଯଦି (sub_mins[j]   > max_min)   max_min   = sub_mins[j];
    }

    ଦଶମିକ inv_scale = 0.0f;
    ଯଦି (max_scale > 0.0f) {
        *d = GGML_FP32_TO_FP16(max_scale / 63.0f);
        ଅଚଳ ଦଶମିକ d_actual = GGML_FP16_TO_FP32(*d);
        inv_scale = (d_actual > 0.0f) ? (1.0f / d_actual) : 0.0f;
    } ଅନ୍ୟଥା {
        *d = GGML_FP32_TO_FP16(0.0f);
    }

    ଦଶମିକ inv_min = 0.0f;
    ଯଦି (max_min > 0.0f) {
        *dmin = GGML_FP32_TO_FP16(max_min / 63.0f);
        ଅଚଳ ଦଶମିକ dmin_actual = GGML_FP16_TO_FP32(*dmin);
        inv_min = (dmin_actual > 0.0f) ? (1.0f / dmin_actual) : 0.0f;
    } ଅନ୍ୟଥା {
        *dmin = GGML_FP32_TO_FP16(0.0f);
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < 8; j++) {
        uint8_t ls = (std::min)((uint8_t) 63, (uint8_t) roundf(inv_scale * sub_scales[j]));
        uint8_t lm = (std::min)((uint8_t) 63, (uint8_t) roundf(inv_min * sub_mins[j]));
        ଯଦି (j < 4) {
            scales[j]     = ls;
            scales[j + 4] = lm;
        } ଅନ୍ୟଥା {
            scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            scales[j - 4] |= ((ls >> 4) << 6);
            scales[j - 0] |= ((lm >> 4) << 6);
        }
    }
}

template <typename BlockT>
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ repack_k4_quants(uint8_t * tile_dst, ଅଚଳ BlockT * b, ପୂର୍ଣ୍ଣସଂଖ୍ୟା row, ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt_local,
                                    int64_t sb_per_row, ତାର୍କିକ use_neon) {
#ଯଦି defined(__ARM_NEON) || defined(__aarch64__)
    ଯଦି (use_neon) {
        ଯଦି ((row & 7) == 0) {
            ଅଚଳ uint8_t * qs[8];
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା rr = 0; rr < 8; ++rr) {
                qs[rr] = b[rr * sb_per_row].qs + (kt_local / 2) * 32;
            }
            repack_neon_k4_rows(qs, tile_dst, row, (kt_local & 1) != 0);
        }
        ଫେରନ୍ତୁ;
    }
#ଅନ୍ୟଥା
    GGML_UNUSED(sb_per_row);
    GGML_UNUSED(use_neon);
#endif
    ଅଚଳ uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା shift = (kt_local & 1) ? 4 : 0;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା cp = 0; cp < 16; cp++) {
        ଅଚଳ uint8_t q0 = (qs_sub[2 * cp + 0] >> shift) & 0x0F;
        ଅଚଳ uint8_t q1 = (qs_sub[2 * cp + 1] >> shift) & 0x0F;
        tile_dst[cp * 32 + row] = (uint8_t) ((q1 << 4) | q0);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ repack_q4_K_tiled(ggml_tensor * t, ଅଚଳ ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    ଅଚଳ int64_t sb_per_row = t->ne[0] / QK_K;
    ଅଚଳ ତାର୍କିକ    use_neon   = (t->ne[1] % 8 == 0);
    repack_kquant_to_tiled_grid<block_q4_K>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_Q4_K,
        [sb_per_row, use_neon](uint8_t * tile_dst, ଅଚଳ block_q4_K * b, ପୂର୍ଣ୍ଣସଂଖ୍ୟା row, ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt_local) {
            ଅଚଳ ଦଶମିକ d    = GGML_FP16_TO_FP32(b->d);
            ଅଚଳ ଦଶମିକ dmin = GGML_FP16_TO_FP32(b->dmin);

            uint8_t sc, m;
            get_scale_min_k4(kt_local, b->scales, &sc, &m);

            repack_k4_quants(tile_dst, b, row, kt_local, sb_per_row, use_neon);

            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
            scale_dst[2 * row + 0] = GGML_FP32_TO_FP16(d * (ଦଶମିକ) sc);
            scale_dst[2 * row + 1] = GGML_FP32_TO_FP16(-dmin * (ଦଶମିକ) m);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ repack_tiled_q4_K(ଶୂନ୍ୟ * data, ଅଚଳ ggml_tensor * t, size_t offset, size_t size) {
    unpack_kquant_from_tiled_grid<block_q4_K>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_Q4_K,
        [](block_q4_K * b, ଅଚଳ uint8_t * sb_tiles, size_t tile_size, ପୂର୍ଣ୍ଣସଂଖ୍ୟା row) {
            ଦଶମିକ sub_scales[8];
            ଦଶମିକ sub_mins[8];

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt_local = 0; kt_local < 8; kt_local++) {
                ଅଚଳ uint8_t *   tile_src  = sb_tiles + kt_local * tile_size;
                ଅଚଳ ggml_half * scale_src = (ଅଚଳ ggml_half *) (tile_src + 512);

                uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା shift  = (kt_local & 1) ? 4 : 0;

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା cp = 0; cp < 16; cp++) {
                    ଅଚଳ uint8_t val = tile_src[cp * 32 + row];
                    qs_sub[2 * cp + 0] |= (uint8_t) ((val & 0x0F) << shift);
                    qs_sub[2 * cp + 1] |= (uint8_t) ((val >> 4)   << shift);
                }

                ଅଚଳ ଦଶମିକ D = GGML_FP16_TO_FP32(scale_src[2 * row + 0]);
                ଅଚଳ ଦଶମିକ M = GGML_FP16_TO_FP32(scale_src[2 * row + 1]);
                sub_scales[kt_local] = (D > 0.0f) ? D : 0.0f;
                sub_mins[kt_local]   = (-M > 0.0f) ? -M : 0.0f;
            }

            pack_k4_scales_mins(b->scales, &b->d, &b->dmin, sub_scales, sub_mins);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ repack_q5_K_tiled(ggml_tensor * t, ଅଚଳ ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    ଅଚଳ int64_t sb_per_row = t->ne[0] / QK_K;
    ଅଚଳ ତାର୍କିକ    use_neon   = (t->ne[1] % 8 == 0);
    repack_kquant_to_tiled_grid<block_q5_K>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_Q5_K,
        [sb_per_row, use_neon](uint8_t * tile_dst, ଅଚଳ block_q5_K * b, ପୂର୍ଣ୍ଣସଂଖ୍ୟା row, ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt_local) {
            ଅଚଳ ଦଶମିକ d     = GGML_FP16_TO_FP32(b->d);
            ଅଚଳ ଦଶମିକ dmin  = GGML_FP16_TO_FP32(b->dmin);
            uint8_t * plane   = tile_dst + 640;

            uint8_t sc, m;
            get_scale_min_k4(kt_local, b->scales, &sc, &m);

            repack_k4_quants(tile_dst, b, row, kt_local, sb_per_row, use_neon);

            ଅଚଳ uint8_t hbit = (uint8_t) (1 << kt_local);
#ଯଦି defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q5_k_qh(b->qh, hbit, plane, row);
#ଅନ୍ୟଥା
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା cp = 0; cp < 16; cp++) {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i    = cp / 4;
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା lane = (cp % 4) * 32 + row;
                ଯଦି (b->qh[2 * cp + 0] & hbit) {
                    plane[lane] |= (uint8_t) (1 << (2 * i));
                }
                ଯଦି (b->qh[2 * cp + 1] & hbit) {
                    plane[lane] |= (uint8_t) (1 << (2 * i + 1));
                }
            }
#endif

            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
            scale_dst[2 * row + 0] = GGML_FP32_TO_FP16(d * (ଦଶମିକ) sc);
            scale_dst[2 * row + 1] = GGML_FP32_TO_FP16(-dmin * (ଦଶମିକ) m);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ repack_tiled_q5_K(ଶୂନ୍ୟ * data, ଅଚଳ ggml_tensor * t, size_t offset, size_t size) {
    unpack_kquant_from_tiled_grid<block_q5_K>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_Q5_K,
        [](block_q5_K * b, ଅଚଳ uint8_t * sb_tiles, size_t tile_size, ପୂର୍ଣ୍ଣସଂଖ୍ୟା row) {
            ଦଶମିକ sub_scales[8];
            ଦଶମିକ sub_mins[8];

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt_local = 0; kt_local < 8; kt_local++) {
                ଅଚଳ uint8_t *   tile_src  = sb_tiles + kt_local * tile_size;
                ଅଚଳ uint8_t *   plane     = tile_src + 640;
                ଅଚଳ ggml_half * scale_src = (ଅଚଳ ggml_half *) (tile_src + 512);

                uint8_t * qs_sub   = b->qs + (kt_local / 2) * 32;
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା shift    = (kt_local & 1) ? 4 : 0;
                ଅଚଳ uint8_t hbit = (uint8_t) (1 << kt_local);

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା cp = 0; cp < 16; cp++) {
                    ଅଚଳ uint8_t val = tile_src[cp * 32 + row];
                    qs_sub[2 * cp + 0] |= (uint8_t) ((val & 0x0F) << shift);
                    qs_sub[2 * cp + 1] |= (uint8_t) ((val >> 4)   << shift);

                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i    = cp / 4;
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା lane = (cp % 4) * 32 + row;
                    ଯଦି (plane[lane] & (1 << (2 * i))) {
                        b->qh[2 * cp + 0] |= hbit;
                    }
                    ଯଦି (plane[lane] & (1 << (2 * i + 1))) {
                        b->qh[2 * cp + 1] |= hbit;
                    }
                }

                ଅଚଳ ଦଶମିକ D = GGML_FP16_TO_FP32(scale_src[2 * row + 0]);
                ଅଚଳ ଦଶମିକ M = GGML_FP16_TO_FP32(scale_src[2 * row + 1]);
                sub_scales[kt_local] = (D > 0.0f) ? D : 0.0f;
                sub_mins[kt_local]   = (-M > 0.0f) ? -M : 0.0f;
            }

            pack_k4_scales_mins(b->scales, &b->d, &b->dmin, sub_scales, sub_mins);
        });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ q1_0_tiled_bit_location(ପୂର୍ଣ୍ଣସଂଖ୍ୟା k, ପୂର୍ଣ୍ଣସଂଖ୍ୟା row, ପୂର୍ଣ୍ଣସଂଖ୍ୟା & byte_offset, ପୂର୍ଣ୍ଣସଂଖ୍ୟା & bit) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା quarter = k / 32;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା kq      = k % 32;

    byte_offset = quarter * 128 + 4 * row + (kq % 4);
    bit         = kq / 4;
}

ସ୍ଥିର ଶୂନ୍ୟ repack_q1_0_tiled(ggml_tensor * t, ଅଚଳ ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    ଅଚଳ block_q1_0 * src_matrix = (ଅଚଳ block_q1_0 *) data;
    ଅଚଳ int64_t ne0 = t->ne[0];
    ଅଚଳ int64_t ne1 = t->ne[1];
    ଅଚଳ int64_t ne2 = t->ne[2];
    ଅଚଳ int64_t ne3 = t->ne[3];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_col_tiles = hex_round_up(ne1, 32) / 32;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_k_tiles = ne0 / QK1_0;
    ଅଚଳ size_t tile_size = HTP_TENSOR_TILE_SIZE_Q1_0;
    ଅଚଳ size_t matrix_size = (size_t) n_col_tiles * n_k_tiles * tile_size;
    ଅଚଳ size_t slice_size = ne1 * ggml_row_size(t->type, ne0);
    ଅଚଳ int64_t start_slice = offset / slice_size;
    ଅଚଳ int64_t end_slice = (std::min)((int64_t) ((offset + size + slice_size - 1) / slice_size), ne2 * ne3);

    GGML_ASSERT(ne0 % QK1_0 == 0);

    ପାଇଁ (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        ଅଚଳ block_q1_0 * src_slice = src_matrix + (slice_idx - start_slice) * ne1 * n_k_tiles;
        uint8_t * matrix_dst = (uint8_t *) t->data + slice_idx * matrix_size;

        repack_parallel_for_ct(matrix_size, n_col_tiles, [&](ପୂର୍ଣ୍ଣସଂଖ୍ୟା ct0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ct1) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ct = ct0; ct < ct1; ct++) {
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = matrix_dst + ((size_t) ct * n_k_tiles + kt) * tile_size;
                    ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
                    memset(tile_dst, 0, tile_size);

                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = 0; row < 32; row++) {
                        ଅଚଳ int64_t r = ct * 32 + row;
                        ଯଦି (r >= ne1) {
                            ଜାରି;
                        }

                        ଅଚଳ block_q1_0 & block = src_slice[r * n_k_tiles + kt];
                        scale_dst[row] = block.d;
#ଯଦି defined(__ARM_NEON) || defined(__aarch64__)
                        repack_neon_q1_0_block(block.qs, tile_dst, row);
#ଅନ୍ୟଥା
                        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < QK1_0; k++) {
                            ପୂର୍ଣ୍ଣସଂଖ୍ୟା byte_offset;
                            ପୂର୍ଣ୍ଣସଂଖ୍ୟା bit;
                            q1_0_tiled_bit_location(k, row, byte_offset, bit);
                            ଯଦି (block.qs[k / 8] & (1u << (k % 8))) {
                                tile_dst[byte_offset] |= (uint8_t) (1u << bit);
                            }
                        }
#endif
                    }
                }
            }
        });
    }
}

ସ୍ଥିର ଶୂନ୍ୟ repack_tiled_q1_0(ଶୂନ୍ୟ * data, ଅଚଳ ggml_tensor * t, size_t offset, size_t size) {
    block_q1_0 * dst_matrix = (block_q1_0 *) data;
    ଅଚଳ int64_t ne0 = t->ne[0];
    ଅଚଳ int64_t ne1 = t->ne[1];
    ଅଚଳ int64_t ne2 = t->ne[2];
    ଅଚଳ int64_t ne3 = t->ne[3];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_col_tiles = hex_round_up(ne1, 32) / 32;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_k_tiles = ne0 / QK1_0;
    ଅଚଳ size_t tile_size = HTP_TENSOR_TILE_SIZE_Q1_0;
    ଅଚଳ size_t matrix_size = (size_t) n_col_tiles * n_k_tiles * tile_size;
    ଅଚଳ size_t row_size_bytes = ggml_row_size(t->type, ne0);
    ଅଚଳ size_t slice_size = ne1 * row_size_bytes;
    ଅଚଳ int64_t start_slice = offset / slice_size;
    ଅଚଳ int64_t end_slice = (std::min)((int64_t) ((offset + size + slice_size - 1) / slice_size), ne2 * ne3);

    GGML_ASSERT(ne0 % QK1_0 == 0);

    ପାଇଁ (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        ଅଚଳ size_t cur_start_byte = (std::max)(offset, (size_t) slice_idx * slice_size);
        ଅଚଳ size_t cur_end_byte = (std::min)(offset + size, (size_t) (slice_idx + 1) * slice_size);
        ଅଚଳ int64_t start_row = (cur_start_byte - (size_t) slice_idx * slice_size) / row_size_bytes;
        ଅଚଳ int64_t end_row = (std::min)((int64_t) ((cur_end_byte - (size_t) slice_idx * slice_size + row_size_bytes - 1) / row_size_bytes), ne1);
        block_q1_0 * dst_slice = dst_matrix + (cur_start_byte - offset) / ଆକାର(block_q1_0);
        ଅଚଳ uint8_t * matrix_src = (ଅଚଳ uint8_t *) t->data + slice_idx * matrix_size;

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ct = start_row / 32; ct < (end_row + 31) / 32; ct++) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା kt = 0; kt < n_k_tiles; kt++) {
                ଅଚଳ uint8_t * tile_src = matrix_src + ((size_t) ct * n_k_tiles + kt) * tile_size;
                ଅଚଳ ggml_half * scale_src = (ଅଚଳ ggml_half *) (tile_src + 512);

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = 0; row < 32; row++) {
                    ଅଚଳ int64_t r = ct * 32 + row;
                    ଯଦି (r < start_row || r >= end_row) {
                        ଜାରି;
                    }

                    block_q1_0 & block = dst_slice[(r - start_row) * n_k_tiles + kt];
                    memset(block.qs, 0, ଆକାର(block.qs));
                    block.d = scale_src[row];
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < QK1_0; k++) {
                        ପୂର୍ଣ୍ଣସଂଖ୍ୟା byte_offset;
                        ପୂର୍ଣ୍ଣସଂଖ୍ୟା bit;
                        q1_0_tiled_bit_location(k, row, byte_offset, bit);
                        ଯଦି (tile_src[byte_offset] & (1u << bit)) {
                            block.qs[k / 8] |= (uint8_t) (1u << (k % 8));
                        }
                    }
                }
            }
        }
    }
}

// --- Top-level Repack / Unpack Dispatchers ---

ଶୂନ୍ୟ htp_repack_tensor_tiled(ggml_tensor * tensor, ଅଚଳ ଶୂନ୍ୟ * data, size_t size) {
    ବାଛ (tensor->type) {
        ସ୍ଥିତି GGML_TYPE_Q1_0:
            repack_q1_0_tiled(tensor, data, 0, size);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q4_0:
        ସ୍ଥିତି GGML_TYPE_IQ4_NL:
            repack_q4_0_tiled(tensor, data, 0, size);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q4_1:
            repack_q4_1_tiled(tensor, data, 0, size);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q8_0:
            repack_q8_0_tiled(tensor, data, 0, size);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_MXFP4:
            repack_mxfp4_tiled(tensor, data, 0, size);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q2_0:
            repack_q2_0_tiled(tensor, data, 0, size);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q2_K:
            repack_q2_K_tiled(tensor, data, 0, size);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q3_K:
            repack_q3_K_tiled(tensor, data, 0, size);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q4_K:
            repack_q4_K_tiled(tensor, data, 0, size);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q5_K:
            repack_q5_K_tiled(tensor, data, 0, size);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q6_K:
            repack_q6_K_tiled(tensor, data, 0, size);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            repack_float_to_f16_tiled<uint16_t>(tensor, data, 0, size,
                [](uint16_t v) { ଫେରନ୍ତୁ v; });
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F32:
            repack_float_to_f16_tiled<ଦଶମିକ>(tensor, data, 0, size,
                [](ଦଶମିକ v) { ଫେରନ୍ତୁ (uint16_t) GGML_FP32_TO_FP16(v); });
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_BF16:
            repack_float_to_f16_tiled<ggml_bf16_t>(tensor, data, 0, size,
                [](ggml_bf16_t v) { ଫେରନ୍ତୁ (uint16_t) GGML_FP32_TO_FP16(GGML_BF16_TO_FP32(v)); });
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଭାଙ୍ଗ;
    }
}

ଶୂନ୍ୟ htp_unpack_tensor_tiled(ଶୂନ୍ୟ * data, ଅଚଳ ggml_tensor * tensor, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor));

    ବାଛ (tensor->type) {
        ସ୍ଥିତି GGML_TYPE_Q1_0:
            repack_tiled_q1_0(data, tensor, offset, size);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q4_0:
        ସ୍ଥିତି GGML_TYPE_IQ4_NL:
            repack_tiled_q4_0(data, tensor, offset, size);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q4_1:
            repack_tiled_q4_1(data, tensor, offset, size);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q8_0:
            repack_tiled_q8_0(data, tensor, offset, size);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_MXFP4:
            repack_tiled_mxfp4(data, tensor, offset, size);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q2_0:
            repack_tiled_q2_0(data, tensor, offset, size);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q2_K:
            repack_tiled_q2_K(data, tensor, offset, size);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q3_K:
            repack_tiled_q3_K(data, tensor, offset, size);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q4_K:
            repack_tiled_q4_K(data, tensor, offset, size);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q5_K:
            repack_tiled_q5_K(data, tensor, offset, size);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q6_K:
            repack_tiled_q6_K(data, tensor, offset, size);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            unpack_f16_tiled_to_float<uint16_t>(data, tensor, offset, size,
                [](uint16_t v) { ଫେରନ୍ତୁ v; });
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F32:
            unpack_f16_tiled_to_float<ଦଶମିକ>(data, tensor, offset, size,
                [](uint16_t v) { ଫେରନ୍ତୁ ggml_fp16_to_fp32((ggml_fp16_t) v); });
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_BF16:
            unpack_f16_tiled_to_float<ggml_bf16_t>(data, tensor, offset, size,
                [](uint16_t v) { ଫେରନ୍ତୁ ggml_fp32_to_bf16(ggml_fp16_to_fp32((ggml_fp16_t) v)); });
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            memcpy(data, (ଅଚଳ ଅକ୍ଷର *) tensor->data + offset, size);
            ଭାଙ୍ଗ;
    }
}
