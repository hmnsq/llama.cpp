#అన్తర్భావ "htp-repack.h"

#నిర్వచించు GGML_COMMON_IMPL_CPP
#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-quants.h"

#అన్తర్భావ <algorithm>
#అన్తర్భావ <cmath>
#అన్తర్భావ <cstring>
#అన్తర్భావ <initializer_list>
#అన్తర్భావ <thread>

#ఒకవేళ defined(__ARM_NEON) || defined(__aarch64__)
#    include <arm_neon.h>
#అయితే_ముగింపు

// --- 4-bit block-32 nibble helpers (shared by Q4_0, IQ4_NL, Q4_1, MXFP4) ---

స్థిర అంతఃవరుస శూన్య unpack_nibbles_32(uint8_t * dst, స్థిరరాశి uint8_t * qs) {
    కొరకు (గుర్తులేని పూర్ణాంకం i = 0; i < 16; ++i) {
        dst[i + 0]  = qs[i] & 0x0F;
        dst[i + 16] = qs[i] >> 4;
    }
}

స్థిర అంతఃవరుస శూన్య pack_nibbles_32(uint8_t * qs, స్థిరరాశి uint8_t * src) {
    కొరకు (గుర్తులేని పూర్ణాంకం i = 0; i < 16; ++i) {
        qs[i] = src[i + 0] | (src[i + 16] << 4);
    }
}

స్థిర అంతఃవరుస శూన్య transpose_nibbles_గమ్యం_tile(uint8_t * tile_dst, స్థిరరాశి uint8_t tile_quants[32][32]) {
    కొరకు (పూర్ణాంకం cp = 0; cp < 16; cp++) {
        కొరకు (పూర్ణాంకం అడ్డువరుస = 0; అడ్డువరుస < 32; అడ్డువరుస++) {
            tile_dst[cp * 32 + అడ్డువరుస] = (tile_quants[అడ్డువరుస][2 * cp + 1] << 4) | tile_quants[అడ్డువరుస][2 * cp];
        }
    }
}

స్థిర అంతఃవరుస శూన్య transpose_tile_గమ్యం_nibbles(uint8_t tile_quants[32][32], స్థిరరాశి uint8_t * tile_src) {
    కొరకు (పూర్ణాంకం cp = 0; cp < 16; cp++) {
        కొరకు (పూర్ణాంకం అడ్డువరుస = 0; అడ్డువరుస < 32; అడ్డువరుస++) {
            స్థిరరాశి uint8_t val = tile_src[cp * 32 + అడ్డువరుస];
            tile_quants[అడ్డువరుస][2 * cp + 0] = val & 0x0F;
            tile_quants[అడ్డువరుస][2 * cp + 1] = val >> 4;
        }
    }
}

#ఒకవేళ defined(__ARM_NEON) || defined(__aarch64__)
స్థిర శూన్య repack_neon_transpose8x8(స్థిరరాశి uint8x8_t అడ్డువరుసలు[8], uint8x8_t columns[8]) {
    స్థిరరాశి uint8x8x2_t t0 = vtrn_u8(అడ్డువరుసలు[0], అడ్డువరుసలు[1]);
    స్థిరరాశి uint8x8x2_t t1 = vtrn_u8(అడ్డువరుసలు[2], అడ్డువరుసలు[3]);
    స్థిరరాశి uint8x8x2_t t2 = vtrn_u8(అడ్డువరుసలు[4], అడ్డువరుసలు[5]);
    స్థిరరాశి uint8x8x2_t t3 = vtrn_u8(అడ్డువరుసలు[6], అడ్డువరుసలు[7]);
    స్థిరరాశి uint16x4x2_t t4 = vtrn_u16(vreinterpret_u16_u8(t0.val[0]), vreinterpret_u16_u8(t1.val[0]));
    స్థిరరాశి uint16x4x2_t t5 = vtrn_u16(vreinterpret_u16_u8(t0.val[1]), vreinterpret_u16_u8(t1.val[1]));
    స్థిరరాశి uint16x4x2_t t6 = vtrn_u16(vreinterpret_u16_u8(t2.val[0]), vreinterpret_u16_u8(t3.val[0]));
    స్థిరరాశి uint16x4x2_t t7 = vtrn_u16(vreinterpret_u16_u8(t2.val[1]), vreinterpret_u16_u8(t3.val[1]));
    స్థిరరాశి uint32x2x2_t t8 = vtrn_u32(vreinterpret_u32_u16(t4.val[0]), vreinterpret_u32_u16(t6.val[0]));
    స్థిరరాశి uint32x2x2_t t9 = vtrn_u32(vreinterpret_u32_u16(t4.val[1]), vreinterpret_u32_u16(t6.val[1]));
    స్థిరరాశి uint32x2x2_t ta = vtrn_u32(vreinterpret_u32_u16(t5.val[0]), vreinterpret_u32_u16(t7.val[0]));
    స్థిరరాశి uint32x2x2_t tb = vtrn_u32(vreinterpret_u32_u16(t5.val[1]), vreinterpret_u32_u16(t7.val[1]));
    columns[0] = vreinterpret_u8_u32(t8.val[0]);
    columns[1] = vreinterpret_u8_u32(ta.val[0]);
    columns[2] = vreinterpret_u8_u32(t9.val[0]);
    columns[3] = vreinterpret_u8_u32(tb.val[0]);
    columns[4] = vreinterpret_u8_u32(t8.val[1]);
    columns[5] = vreinterpret_u8_u32(ta.val[1]);
    columns[6] = vreinterpret_u8_u32(t9.val[1]);
    columns[7] = vreinterpret_u8_u32(tb.val[1]);
}

స్థిర uint8x8_t repack_neon_pack_4bit_జతలు(uint8x8_t మొదటి, uint8x8_t second, తార్కిక ఎగువ) {
    స్థిరరాశి uint8x8x2_t separated = vuzp_u8(మొదటి, second);
    స్థిరరాశి uint8x8_t ముసుగు = vdup_n_u8(0x0F);
    uint8x8_t even = separated.val[0];
    uint8x8_t odd  = separated.val[1];

    ఒకవేళ (ఎగువ) {
        even = vshr_n_u8(even, 4);
        odd  = vshr_n_u8(odd, 4);
    }

    even = vand_u8(even, ముసుగు);
    odd  = vand_u8(odd, ముసుగు);
    తిరిగి vorr_u8(vshl_n_u8(odd, 4), even);
}

స్థిర శూన్య repack_neon_4bit_అడ్డువరుసలు(స్థిరరాశి uint8_t * qs[8], uint8_t * tile_dst, పూర్ణాంకం row0) {
    uint8x8_t దిగువ_అడ్డువరుసలు[8];
    uint8x8_t ఎగువ_అడ్డువరుసలు[8];
    uint8x8_t columns[8];

    కొరకు (పూర్ణాంకం rr = 0; rr < 8; ++rr) {
        స్థిరరాశి uint8x16_t packed = vld1q_u8(qs[rr]);
        స్థిరరాశి uint8x8_t మొదటి = vget_దిగువ_u8(packed);
        స్థిరరాశి uint8x8_t second = vget_ఎగువ_u8(packed);
        దిగువ_అడ్డువరుసలు[rr]  = repack_neon_pack_4bit_జతలు(మొదటి, second, అబద్ధం);
        ఎగువ_అడ్డువరుసలు[rr] = repack_neon_pack_4bit_జతలు(మొదటి, second, నిజం);
    }

    repack_neon_transpose8x8(దిగువ_అడ్డువరుసలు, columns);
    కొరకు (పూర్ణాంకం cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + cp * 32 + row0, columns[cp]);
    }

    repack_neon_transpose8x8(ఎగువ_అడ్డువరుసలు, columns);
    కొరకు (పూర్ణాంకం cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + (cp + 8) * 32 + row0, columns[cp]);
    }
}

స్థిర శూన్య repack_neon_k4_అడ్డువరుసలు(స్థిరరాశి uint8_t * qs[8], uint8_t * tile_dst, పూర్ణాంకం row0, తార్కిక ఎగువ) {
    uint8x8_t మొదటి_అడ్డువరుసలు[8];
    uint8x8_t రెండవ_అడ్డువరుసలు[8];
    uint8x8_t columns[8];

    కొరకు (పూర్ణాంకం rr = 0; rr < 8; ++rr) {
        మొదటి_అడ్డువరుసలు[rr] = repack_neon_pack_4bit_జతలు(vld1_u8(qs[rr]), vld1_u8(qs[rr] + 8), ఎగువ);
        రెండవ_అడ్డువరుసలు[rr] = repack_neon_pack_4bit_జతలు(vld1_u8(qs[rr] + 16), vld1_u8(qs[rr] + 24), ఎగువ);
    }

    repack_neon_transpose8x8(మొదటి_అడ్డువరుసలు, columns);
    కొరకు (పూర్ణాంకం cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + cp * 32 + row0, columns[cp]);
    }

    repack_neon_transpose8x8(రెండవ_అడ్డువరుసలు, columns);
    కొరకు (పూర్ణాంకం cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + (cp + 8) * 32 + row0, columns[cp]);
    }
}

స్థిర శూన్య repack_neon_q8_అడ్డువరుసలు(స్థిరరాశి uint8_t * qs[8], uint8_t * tile_dst, పూర్ణాంకం row0) {
    uint8x8_t అడ్డువరుసలు[8];
    uint8x8_t columns[8];

    కొరకు (పూర్ణాంకం cp = 0; cp < 16; cp += 4) {
        కొరకు (పూర్ణాంకం rr = 0; rr < 8; ++rr) {
            అడ్డువరుసలు[rr] = vld1_u8(qs[rr] + 2 * cp);
        }
        repack_neon_transpose8x8(అడ్డువరుసలు, columns);

        కొరకు (పూర్ణాంకం pair = 0; pair < 4; ++pair) {
            స్థిరరాశి uint8x8x2_t interleaved = vzip_u8(columns[2 * pair], columns[2 * pair + 1]);
            uint8_t * dst = tile_dst + (cp + pair) * 64 + 2 * row0;
            vst1_u8(dst, interleaved.val[0]);
            vst1_u8(dst + 8, interleaved.val[1]);
        }
    }
}

స్థిర అంతఃవరుస శూన్య repack_store_u32(uint8_t * dst, uint32_t v) {
    జ్ఞాపకం_నకలు(dst, &v, పరిమాణం(v));
}

స్థిర అంతఃవరుస uint32x4_t repack_neon_transpose_2bit_4x4(uint32x4_t w) {
    uint32x4_t t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 6)), vdupq_n_u32(0x00CC00CCu));
    w = veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 6)));
    t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 12)), vdupq_n_u32(0x0000F0F0u));
    తిరిగి veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 12)));
}

స్థిర అంతఃవరుస శూన్య repack_neon_q2_0_ఖండం(స్థిరరాశి uint8_t * qs, uint8_t * tile, పూర్ణాంకం అడ్డువరుస) {
    స్థిరరాశి uint32x4_t w = repack_neon_transpose_2bit_4x4(vreinterpretq_u32_u8(vld1q_u8(qs)));
    repack_store_u32(tile + 0 * 128 + 4 * అడ్డువరుస, vgetq_lane_u32(w, 0));
    repack_store_u32(tile + 1 * 128 + 4 * అడ్డువరుస, vgetq_lane_u32(w, 1));
    repack_store_u32(tile + 2 * 128 + 4 * అడ్డువరుస, vgetq_lane_u32(w, 2));
    repack_store_u32(tile + 3 * 128 + 4 * అడ్డువరుస, vgetq_lane_u32(w, 3));
}

స్థిర అంతఃవరుస శూన్య repack_neon_q1_0_ఖండం(స్థిరరాశి uint8_t * qs, uint8_t * tile, పూర్ణాంకం అడ్డువరుస) {
    uint32x4_t w = vreinterpretq_u32_u8(vld1q_u8(qs));
    uint32x4_t t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 1)), vdupq_n_u32(0x22222222u));
    w = veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 1)));
    t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 3)), vdupq_n_u32(0x0A0A0A0Au));
    w = veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 3)));
    w = repack_neon_transpose_2bit_4x4(w);
    repack_store_u32(tile + 0 * 128 + 4 * అడ్డువరుస, vgetq_lane_u32(w, 0));
    repack_store_u32(tile + 1 * 128 + 4 * అడ్డువరుస, vgetq_lane_u32(w, 1));
    repack_store_u32(tile + 2 * 128 + 4 * అడ్డువరుస, vgetq_lane_u32(w, 2));
    repack_store_u32(tile + 3 * 128 + 4 * అడ్డువరుస, vgetq_lane_u32(w, 3));
}

స్థిర అంతఃవరుస శూన్య repack_neon_pack_2bit_plane(uint8x16_t v0, uint8x16_t v1, uint8_t * dst, పూర్ణాంకం అడ్డువరుస) {
    స్థిరరాశి uint32x4x2_t z   = vuzpq_u32(vreinterpretq_u32_u8(v0), vreinterpretq_u32_u8(v1));
    స్థిరరాశి uint32x4_t   p01 = vorrq_u32(z.val[0], vshlq_n_u32(z.val[1], 2));
    స్థిరరాశి uint32x2x2_t u   = vuzp_u32(vget_దిగువ_u32(p01), vget_ఎగువ_u32(p01));
    స్థిరరాశి uint32x2_t   బహిర్గతం = vorr_u32(u.val[0], vshl_n_u32(u.val[1], 4));
    repack_store_u32(dst + 0 * 128 + అడ్డువరుస * 4, vget_lane_u32(బహిర్గతం, 0));
    repack_store_u32(dst + 1 * 128 + అడ్డువరుస * 4, vget_lane_u32(బహిర్గతం, 1));
}

స్థిర అంతఃవరుస శూన్య repack_neon_q2_3_k_low2(స్థిరరాశి uint8_t * qs, పూర్ణాంకం kt_local, uint8_t * lo_pl, పూర్ణాంకం అడ్డువరుస) {
    స్థిరరాశి uint8_t *  qs_sub = qs + (kt_local / 4) * 32;
    స్థిరరాశి int8x16_t  జరుపు  = vdupq_n_s8(-2 * (kt_local % 4));
    స్థిరరాశి uint8x16_t ముసుగు   = vdupq_n_u8(0x03);
    స్థిరరాశి uint8x16_t v0     = vandq_u8(vshlq_u8(vld1q_u8(qs_sub +  0), జరుపు), ముసుగు);
    స్థిరరాశి uint8x16_t v1     = vandq_u8(vshlq_u8(vld1q_u8(qs_sub + 16), జరుపు), ముసుగు);
    repack_neon_pack_2bit_plane(v0, v1, lo_pl, అడ్డువరుస);
}

స్థిర అంతఃవరుస శూన్య repack_neon_q3_k_hmask(స్థిరరాశి uint8_t * hmask, పూర్ణాంకం kt_local, uint8_t * neg_pl, పూర్ణాంకం అడ్డువరుస) {
    స్థిర స్థిరరాశి uint32_t ముసుగు_పదాలు[4] = { 0x01010101u, 0x02020202u, 0x04040404u, 0x08080808u };
    స్థిరరాశి uint8x16_t m0   = vreinterpretq_u8_u32(vld1q_u32(ముసుగు_పదాలు));
    స్థిరరాశి uint8x16_t m1   = vshlq_n_u8(m0, 4);
    స్థిరరాశి uint8x16_t hbit = vdupq_n_u8((uint8_t) (1u << kt_local));
    స్థిరరాశి uint8x16_t b0   = vbicq_u8(m0, vtstq_u8(vld1q_u8(hmask +  0), hbit));
    స్థిరరాశి uint8x16_t b1   = vbicq_u8(m1, vtstq_u8(vld1q_u8(hmask + 16), hbit));
    స్థిరరాశి uint32x4_t b01  = vreinterpretq_u32_u8(vorrq_u8(b0, b1));
    స్థిరరాశి uint32x2_t f0   = vorr_u32(vget_దిగువ_u32(b01), vget_ఎగువ_u32(b01));
    స్థిరరాశి uint32x2_t f1   = vpadd_u32(f0, f0);
    repack_store_u32(neg_pl + అడ్డువరుస * 4, vget_lane_u32(f1, 0));
}

స్థిర అంతఃవరుస శూన్య repack_neon_q6_k(స్థిరరాశి ఖండం_q6_K * b, పూర్ణాంకం kt_local, uint8_t * lo_pl, uint8_t * hi_pl, పూర్ణాంకం అడ్డువరుస) {
    స్థిరరాశి పూర్ణాంకం        c      = kt_local / 4;
    స్థిరరాశి పూర్ణాంకం        g_sb   = kt_local % 4;
    స్థిరరాశి uint8_t *  ql     = b->ql + c * 64 + (g_sb & 1) * 32;
    స్థిరరాశి uint8_t *  qh     = b->qh + c * 32;
    స్థిరరాశి int8x16_t  lshift = vdupq_n_s8((g_sb >= 2) ? -4 : 0);
    స్థిరరాశి int8x16_t  hshift = vdupq_n_s8(-2 * g_sb);
    స్థిరరాశి uint8x16_t mask4  = vdupq_n_u8(0x0F);
    స్థిరరాశి uint8x16_t mask2  = vdupq_n_u8(0x03);

    స్థిరరాశి uint8x16_t lo0  = vandq_u8(vshlq_u8(vld1q_u8(ql +  0), lshift), mask4);
    స్థిరరాశి uint8x16_t lo1  = vandq_u8(vshlq_u8(vld1q_u8(ql + 16), lshift), mask4);
    స్థిరరాశి uint32x4x2_t lz = vuzpq_u32(vreinterpretq_u32_u8(lo0), vreinterpretq_u32_u8(lo1));
    స్థిరరాశి uint32x4_t lout = vorrq_u32(lz.val[0], vshlq_n_u32(lz.val[1], 4));
    repack_store_u32(lo_pl + 0 * 128 + అడ్డువరుస * 4, vgetq_lane_u32(lout, 0));
    repack_store_u32(lo_pl + 1 * 128 + అడ్డువరుస * 4, vgetq_lane_u32(lout, 1));
    repack_store_u32(lo_pl + 2 * 128 + అడ్డువరుస * 4, vgetq_lane_u32(lout, 2));
    repack_store_u32(lo_pl + 3 * 128 + అడ్డువరుస * 4, vgetq_lane_u32(lout, 3));

    స్థిరరాశి uint8x16_t hi0 = vandq_u8(vshlq_u8(vld1q_u8(qh +  0), hshift), mask2);
    స్థిరరాశి uint8x16_t hi1 = vandq_u8(vshlq_u8(vld1q_u8(qh + 16), hshift), mask2);
    repack_neon_pack_2bit_plane(hi0, hi1, hi_pl, అడ్డువరుస);
}

స్థిర అంతఃవరుస శూన్య repack_neon_q5_k_qh(స్థిరరాశి uint8_t * qh, uint8_t hbit, uint8_t * plane, పూర్ణాంకం అడ్డువరుస) {
    స్థిర స్థిరరాశి uint8_t ముసుగు_బైట్లు[16] = { 1, 2, 1, 2, 1, 2, 1, 2, 4, 8, 4, 8, 4, 8, 4, 8 };
    స్థిరరాశి uint8x16_t m0   = vld1q_u8(ముసుగు_బైట్లు);
    స్థిరరాశి uint8x16_t m1   = vshlq_n_u8(m0, 4);
    స్థిరరాశి uint8x16_t vbit = vdupq_n_u8(hbit);
    స్థిరరాశి uint8x16_t b0   = vandq_u8(vtstq_u8(vld1q_u8(qh +  0), vbit), m0);
    స్థిరరాశి uint8x16_t b1   = vandq_u8(vtstq_u8(vld1q_u8(qh + 16), vbit), m1);
    స్థిరరాశి uint8x16_t b01  = vorrq_u8(b0, b1);
    స్థిరరాశి uint8x8_t  fold = vorr_u8(vget_దిగువ_u8(b01), vget_ఎగువ_u8(b01));
    స్థిరరాశి uint8x8_t  బహిర్గతం  = vpadd_u8(fold, fold);
    plane[0 * 32 + అడ్డువరుస] = vget_lane_u8(బహిర్గతం, 0);
    plane[1 * 32 + అడ్డువరుస] = vget_lane_u8(బహిర్గతం, 1);
    plane[2 * 32 + అడ్డువరుస] = vget_lane_u8(బహిర్గతం, 2);
    plane[3 * 32 + అడ్డువరుస] = vget_lane_u8(బహిర్గతం, 3);
}

స్థిర అంతఃవరుస uint16x8_t repack_neon_ఎక్కించు_f16x8(స్థిరరాశి uint16_t * src) {
    తిరిగి vld1q_u16(src);
}

#ఒకవేళ defined(__aarch64__) || defined(__ARM_FP16_FORMAT_IEEE)
స్థిర అంతఃవరుస uint16x8_t repack_neon_ఎక్కించు_f16x8(స్థిరరాశి దశాంశం * src) {
    స్థిరరాశి float16x4_t lo = vcvt_f16_f32(vld1q_f32(src));
    స్థిరరాశి float16x4_t hi = vcvt_f16_f32(vld1q_f32(src + 4));
    తిరిగి vreinterpretq_u16_f16(vcombine_f16(lo, hi));
}

స్థిర అంతఃవరుస uint16x8_t repack_neon_ఎక్కించు_f16x8(స్థిరరాశి ggml_bf16_t * src) {
    స్థిరరాశి uint16x8_t  bf   = vld1q_u16((స్థిరరాశి uint16_t *) src);
    స్థిరరాశి float32x4_t f_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_దిగువ_u16(bf), 16));
    స్థిరరాశి float32x4_t f_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_ఎగువ_u16(bf), 16));
    స్థిరరాశి float16x4_t lo   = vcvt_f16_f32(f_lo);
    స్థిరరాశి float16x4_t hi   = vcvt_f16_f32(f_hi);
    తిరిగి vreinterpretq_u16_f16(vcombine_f16(lo, hi));
}
#లేకపోతే
స్థిర అంతఃవరుస uint16x8_t repack_neon_ఎక్కించు_f16x8(స్థిరరాశి దశాంశం * src) {
    uint16_t tmp[8];
    కొరకు (పూర్ణాంకం i = 0; i < 8; ++i) {
        tmp[i] = GGML_FP32_గమ్యం_FP16(src[i]);
    }
    తిరిగి vld1q_u16(tmp);
}

స్థిర అంతఃవరుస uint16x8_t repack_neon_ఎక్కించు_f16x8(స్థిరరాశి ggml_bf16_t * src) {
    uint16_t tmp[8];
    కొరకు (పూర్ణాంకం i = 0; i < 8; ++i) {
        tmp[i] = GGML_FP32_గమ్యం_FP16(GGML_BF16_గమ్యం_FP32(src[i]));
    }
    తిరిగి vld1q_u16(tmp);
}
#అయితే_ముగింపు

మూస <రకం_పేరు SrcT>
స్థిర శూన్య repack_neon_f16_tile(uint16_t * tile_dst, స్థిరరాశి SrcT * src_ముక్క, int64_t అడ్డువరుస_stride, పూర్ణాంకం ct, పూర్ణాంకం kt) {
    కొరకు (పూర్ణాంకం row0 = 0; row0 < 32; row0 += 4) {
        స్థిరరాశి SrcT * r0 = src_ముక్క + (int64_t) (ct * 32 + row0 + 0) * అడ్డువరుస_stride + kt * 32;
        స్థిరరాశి SrcT * r1 = src_ముక్క + (int64_t) (ct * 32 + row0 + 1) * అడ్డువరుస_stride + kt * 32;
        స్థిరరాశి SrcT * r2 = src_ముక్క + (int64_t) (ct * 32 + row0 + 2) * అడ్డువరుస_stride + kt * 32;
        స్థిరరాశి SrcT * r3 = src_ముక్క + (int64_t) (ct * 32 + row0 + 3) * అడ్డువరుస_stride + kt * 32;
        కొరకు (పూర్ణాంకం cp = 0; cp < 16; cp += 4) {
            స్థిరరాశి uint32x4_t v0 = vreinterpretq_u32_u16(repack_neon_ఎక్కించు_f16x8(r0 + 2 * cp));
            స్థిరరాశి uint32x4_t v1 = vreinterpretq_u32_u16(repack_neon_ఎక్కించు_f16x8(r1 + 2 * cp));
            స్థిరరాశి uint32x4_t v2 = vreinterpretq_u32_u16(repack_neon_ఎక్కించు_f16x8(r2 + 2 * cp));
            స్థిరరాశి uint32x4_t v3 = vreinterpretq_u32_u16(repack_neon_ఎక్కించు_f16x8(r3 + 2 * cp));

            స్థిరరాశి uint32x2x2_t t0 = vtrn_u32(vget_దిగువ_u32(v0),  vget_దిగువ_u32(v1));
            స్థిరరాశి uint32x2x2_t t1 = vtrn_u32(vget_ఎగువ_u32(v0), vget_ఎగువ_u32(v1));
            స్థిరరాశి uint32x2x2_t t2 = vtrn_u32(vget_దిగువ_u32(v2),  vget_దిగువ_u32(v3));
            స్థిరరాశి uint32x2x2_t t3 = vtrn_u32(vget_ఎగువ_u32(v2), vget_ఎగువ_u32(v3));

            vst1q_u16(tile_dst + (cp + 0) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t0.val[0], t2.val[0])));
            vst1q_u16(tile_dst + (cp + 1) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t0.val[1], t2.val[1])));
            vst1q_u16(tile_dst + (cp + 2) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t1.val[0], t3.val[0])));
            vst1q_u16(tile_dst + (cp + 3) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t1.val[1], t3.val[1])));
        }
    }
}
#అయితే_ముగింపు

// --- Generic 32x32 tile grid walkers for block-32 and float types ---

మూస <రకం_పేరు Fn>
స్థిర శూన్య repack_parallel_for_ct(size_t ముక్క_బైట్లు, పూర్ణాంకం n_నిలువు_tiles, Fn && fn) {
    పూర్ణాంకం n_దారాలు = 1;
    ఒకవేళ (ముక్క_బైట్లు > 128 * 1024) {
        కొరకు (పూర్ణాంకం nt : { 4, 3, 2 }) {
            ఒకవేళ (n_నిలువు_tiles % nt == 0) {
                n_దారాలు = nt;
                ఆపు;
            }
        }
    }

    ఒకవేళ (n_దారాలు <= 1) {
        fn(0, n_నిలువు_tiles);
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం chunk = n_నిలువు_tiles / n_దారాలు;
    std::thread workers[3];
    కొరకు (పూర్ణాంకం i = 0; i < n_దారాలు - 1; ++i) {
        workers[i] = std::thread([&fn, i, chunk]() {
            fn(i * chunk, (i + 1) * chunk);
        });
    }
    fn((n_దారాలు - 1) * chunk, n_నిలువు_tiles);
    కొరకు (పూర్ణాంకం i = 0; i < n_దారాలు - 1; ++i) {
        workers[i].join();
    }
}

మూస <రకం_పేరు SrcElemT, పూర్ణాంకం ELEMS_చొప్పున_KTILE, రకం_పేరు PackTileFn>
స్థిర శూన్య repack_గమ్యం_tiled_grid_32(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size,
                                    size_t tile_కొలత, PackTileFn && pack_tile) {
    స్థిరరాశి SrcElemT * src_మాతృక = (స్థిరరాశి SrcElemT *) దత్తాంశం;
    స్థిరరాశి int64_t ne0 = t->ne[0];
    స్థిరరాశి int64_t ne1 = t->ne[1];
    స్థిరరాశి int64_t ne2 = t->ne[2];
    స్థిరరాశి int64_t ne3 = t->ne[3];
    స్థిరరాశి పూర్ణాంకం n_నిలువు_tiles = hex_విడత_పైకి(ne1, 32) / 32;
    స్థిరరాశి పూర్ణాంకం n_k_tiles   = hex_విడత_పైకి(ne0, 32) / 32;
    స్థిరరాశి size_t మాతృక_కొలత = (size_t) n_నిలువు_tiles * n_k_tiles * tile_కొలత;

    స్థిరరాశి size_t ముక్క_కొలత = (size_t) ne1 * ggml_అడ్డువరుస_కొలత(t->type, ne0);
    స్థిరరాశి int64_t ప్రారంభం_ముక్క = స్థానభ్రంశం / ముక్క_కొలత;
    int64_t చివర_ముక్క = (స్థానభ్రంశం + size + ముక్క_కొలత - 1) / ముక్క_కొలత;
    ఒకవేళ (చివర_ముక్క > ne2 * ne3) {
        చివర_ముక్క = ne2 * ne3;
    }

    స్థిరరాశి int64_t అడ్డువరుస_stride = (ne0 / (32 / ELEMS_చొప్పున_KTILE));

    కొరకు (int64_t ముక్క_idx = ప్రారంభం_ముక్క; ముక్క_idx < చివర_ముక్క; ముక్క_idx++) {
        స్థిరరాశి SrcElemT * src_ముక్క = src_మాతృక + (ముక్క_idx - ప్రారంభం_ముక్క) * (ne1 * అడ్డువరుస_stride);
        uint8_t * మాతృక_dst = (uint8_t *) t->దత్తాంశం + ముక్క_idx * మాతృక_కొలత;

        repack_parallel_for_ct(మాతృక_కొలత, n_నిలువు_tiles, [&](పూర్ణాంకం ct0, పూర్ణాంకం ct1) {
            కొరకు (పూర్ణాంకం ct = ct0; ct < ct1; ct++) {
                కొరకు (పూర్ణాంకం kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = మాతృక_dst + ((size_t) ct * n_k_tiles + kt) * tile_కొలత;
                    pack_tile(tile_dst, src_ముక్క, ne0, ne1, అడ్డువరుస_stride, ct, kt);
                }
            }
        });
    }
}

మూస <రకం_పేరు DstElemT, పూర్ణాంకం ELEMS_చొప్పున_KTILE, రకం_పేరు UnpackTileFn>
స్థిర శూన్య unpack_from_tiled_grid_32(శూన్య * దత్తాంశం, స్థిరరాశి ggml_tensor * t, size_t స్థానభ్రంశం, size_t size,
                                      size_t tile_కొలత, UnpackTileFn && unpack_tile) {
    uint8_t * dst_బైట్లు = (uint8_t *) దత్తాంశం;
    స్థిరరాశి int64_t ne0 = t->ne[0];
    స్థిరరాశి int64_t ne1 = t->ne[1];
    స్థిరరాశి int64_t ne2 = t->ne[2];
    స్థిరరాశి int64_t ne3 = t->ne[3];
    స్థిరరాశి పూర్ణాంకం n_నిలువు_tiles = hex_విడత_పైకి(ne1, 32) / 32;
    స్థిరరాశి పూర్ణాంకం n_k_tiles   = hex_విడత_పైకి(ne0, 32) / 32;
    స్థిరరాశి size_t మాతృక_కొలత = (size_t) n_నిలువు_tiles * n_k_tiles * tile_కొలత;

    స్థిరరాశి size_t అడ్డువరుస_కొలత_బైట్లు = ggml_అడ్డువరుస_కొలత(t->type, ne0);
    స్థిరరాశి size_t ముక్క_కొలత     = (size_t) ne1 * అడ్డువరుస_కొలత_బైట్లు;
    స్థిరరాశి int64_t ప్రారంభం_ముక్క   = స్థానభ్రంశం / ముక్క_కొలత;
    int64_t చివర_ముక్క           = (స్థానభ్రంశం + size + ముక్క_కొలత - 1) / ముక్క_కొలత;
    ఒకవేళ (చివర_ముక్క > ne2 * ne3) {
        చివర_ముక్క = ne2 * ne3;
    }

    స్థిరరాశి int64_t అడ్డువరుస_stride = (ne0 / (32 / ELEMS_చొప్పున_KTILE));

    కొరకు (int64_t ముక్క_idx = ప్రారంభం_ముక్క; ముక్క_idx < చివర_ముక్క; ముక్క_idx++) {
        స్థిరరాశి size_t cur_ప్రారంభం_byte     = (std::max)(స్థానభ్రంశం, (size_t) ముక్క_idx * ముక్క_కొలత);
        స్థిరరాశి size_t cur_చివర_byte       = (std::min)(స్థానభ్రంశం + size, (size_t) (ముక్క_idx + 1) * ముక్క_కొలత);
        స్థిరరాశి size_t ముక్క_స్థానభ్రంశం_ప్రారంభం = cur_ప్రారంభం_byte - (size_t) ముక్క_idx * ముక్క_కొలత;
        స్థిరరాశి size_t ముక్క_స్థానభ్రంశం_చివర   = cur_చివర_byte - (size_t) ముక్క_idx * ముక్క_కొలత;

        స్థిరరాశి int64_t ప్రారంభం_అడ్డువరుస = ముక్క_స్థానభ్రంశం_ప్రారంభం / అడ్డువరుస_కొలత_బైట్లు;
        స్థిరరాశి int64_t చివర_అడ్డువరుస   = (std::min)((int64_t) ((ముక్క_స్థానభ్రంశం_చివర + అడ్డువరుస_కొలత_బైట్లు - 1) / అడ్డువరుస_కొలత_బైట్లు), ne1);

        స్థిరరాశి పూర్ణాంకం ప్రారంభం_ct = (పూర్ణాంకం) (ప్రారంభం_అడ్డువరుస / 32);
        స్థిరరాశి పూర్ణాంకం చివర_ct   = (std::min)((పూర్ణాంకం) ((చివర_అడ్డువరుస + 31) / 32), n_నిలువు_tiles);

        DstElemT * dst_ముక్క = (DstElemT *) (dst_బైట్లు + (cur_ప్రారంభం_byte - స్థానభ్రంశం));
        స్థిరరాశి uint8_t * మాతృక_src = (స్థిరరాశి uint8_t *) t->దత్తాంశం + ముక్క_idx * మాతృక_కొలత;

        కొరకు (పూర్ణాంకం ct = ప్రారంభం_ct; ct < చివర_ct; ct++) {
            కొరకు (పూర్ణాంకం kt = 0; kt < n_k_tiles; kt++) {
                స్థిరరాశి uint8_t * tile_src = మాతృక_src + ((size_t) ct * n_k_tiles + kt) * tile_కొలత;
                unpack_tile(dst_ముక్క, tile_src, ne0, ప్రారంభం_అడ్డువరుస, చివర_అడ్డువరుస, అడ్డువరుస_stride, ct, kt);
            }
        }
    }
}

// --- Block-32 Quantized Types (Q4_0, IQ4_NL, Q4_1, Q8_0, MXFP4) ---

స్థిర శూన్య repack_q4_0_tiled(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    repack_గమ్యం_tiled_grid_32<ఖండం_q4_0, 1>(t, దత్తాంశం, స్థానభ్రంశం, size, HTP_TENSOR_TILE_కొలత_Q4_0,
        [](uint8_t * tile_dst, స్థిరరాశి ఖండం_q4_0 * src_ముక్క, int64_t ne0, int64_t ne1, int64_t అడ్డువరుస_stride, పూర్ణాంకం ct, పూర్ణాంకం kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
#ఒకవేళ defined(__ARM_NEON) || defined(__aarch64__)
            ఒకవేళ (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                కొరకు (పూర్ణాంకం row0 = 0; row0 < 32; row0 += 8) {
                    స్థిరరాశి uint8_t * qs[8];
                    కొరకు (పూర్ణాంకం rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_ముక్క[(ct * 32 + row0 + rr) * అడ్డువరుస_stride + kt].qs;
                    }
                    repack_neon_4bit_అడ్డువరుసలు(qs, tile_dst, row0);
                }

                కొరకు (పూర్ణాంకం అడ్డువరుస = 0; అడ్డువరుస < 32; ++అడ్డువరుస) {
                    scale_dst[అడ్డువరుస] = src_ముక్క[(ct * 32 + అడ్డువరుస) * అడ్డువరుస_stride + kt].d;
                }
                తిరిగి;
            }
#అయితే_ముగింపు
            uint8_t tile_quants[32][32];
            కొరకు (పూర్ణాంకం అడ్డువరుస = 0; అడ్డువరుస < 32; అడ్డువరుస++) {
                స్థిరరాశి int64_t r = ct * 32 + అడ్డువరుస;
                ఒకవేళ (r < ne1 && kt < ne0 / 32) {
                    స్థిరరాశి ఖండం_q4_0 & b = src_ముక్క[r * అడ్డువరుస_stride + kt];
                    unpack_nibbles_32(tile_quants[అడ్డువరుస], b.qs);
                    scale_dst[అడ్డువరుస] = b.d;
                } లేకపోతే {
                    memset(tile_quants[అడ్డువరుస], 8, 32);
                    scale_dst[అడ్డువరుస] = 0;
                }
            }
            transpose_nibbles_గమ్యం_tile(tile_dst, tile_quants);
        });
}

స్థిర శూన్య repack_tiled_q4_0(శూన్య * దత్తాంశం, స్థిరరాశి ggml_tensor * t, size_t స్థానభ్రంశం, size_t size) {
    unpack_from_tiled_grid_32<ఖండం_q4_0, 1>(దత్తాంశం, t, స్థానభ్రంశం, size, HTP_TENSOR_TILE_కొలత_Q4_0,
        [](ఖండం_q4_0 * dst_ముక్క, స్థిరరాశి uint8_t * tile_src, int64_t ne0, int64_t ప్రారంభం_అడ్డువరుస, int64_t చివర_అడ్డువరుస, int64_t అడ్డువరుస_stride, పూర్ణాంకం ct, పూర్ణాంకం kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_గమ్యం_nibbles(tile_quants, tile_src);
            స్థిరరాశి ggml_half * scale_src = (స్థిరరాశి ggml_half *) (tile_src + 512);
            కొరకు (పూర్ణాంకం అడ్డువరుస = 0; అడ్డువరుస < 32; అడ్డువరుస++) {
                స్థిరరాశి int64_t r = ct * 32 + అడ్డువరుస;
                ఒకవేళ (r >= ప్రారంభం_అడ్డువరుస && r < చివర_అడ్డువరుస && kt < ne0 / 32) {
                    ఖండం_q4_0 & b = dst_ముక్క[(r - ప్రారంభం_అడ్డువరుస) * అడ్డువరుస_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[అడ్డువరుస]);
                    b.d = scale_src[అడ్డువరుస];
                }
            }
        });
}

స్థిర శూన్య repack_q4_1_tiled(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    repack_గమ్యం_tiled_grid_32<ఖండం_q4_1, 1>(t, దత్తాంశం, స్థానభ్రంశం, size, HTP_TENSOR_TILE_కొలత_Q4_1,
        [](uint8_t * tile_dst, స్థిరరాశి ఖండం_q4_1 * src_ముక్క, int64_t ne0, int64_t ne1, int64_t అడ్డువరుస_stride, పూర్ణాంకం ct, పూర్ణాంకం kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
#ఒకవేళ defined(__ARM_NEON) || defined(__aarch64__)
            ఒకవేళ (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                కొరకు (పూర్ణాంకం row0 = 0; row0 < 32; row0 += 8) {
                    స్థిరరాశి uint8_t * qs[8];
                    కొరకు (పూర్ణాంకం rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_ముక్క[(ct * 32 + row0 + rr) * అడ్డువరుస_stride + kt].qs;
                    }
                    repack_neon_4bit_అడ్డువరుసలు(qs, tile_dst, row0);
                }

                కొరకు (పూర్ణాంకం అడ్డువరుస = 0; అడ్డువరుస < 32; ++అడ్డువరుస) {
                    స్థిరరాశి ఖండం_q4_1 & src_ఖండం = src_ముక్క[(ct * 32 + అడ్డువరుస) * అడ్డువరుస_stride + kt];
                    scale_dst[2 * అడ్డువరుస + 0] = src_ఖండం.d;
                    scale_dst[2 * అడ్డువరుస + 1] = src_ఖండం.m;
                }
                తిరిగి;
            }
#అయితే_ముగింపు
            uint8_t tile_quants[32][32];
            కొరకు (పూర్ణాంకం అడ్డువరుస = 0; అడ్డువరుస < 32; అడ్డువరుస++) {
                స్థిరరాశి int64_t r = ct * 32 + అడ్డువరుస;
                ఒకవేళ (r < ne1 && kt < ne0 / 32) {
                    స్థిరరాశి ఖండం_q4_1 & b = src_ముక్క[r * అడ్డువరుస_stride + kt];
                    unpack_nibbles_32(tile_quants[అడ్డువరుస], b.qs);
                    scale_dst[2 * అడ్డువరుస + 0] = b.d;
                    scale_dst[2 * అడ్డువరుస + 1] = b.m;
                } లేకపోతే {
                    memset(tile_quants[అడ్డువరుస], 0, 32);
                    scale_dst[2 * అడ్డువరుస + 0] = 0;
                    scale_dst[2 * అడ్డువరుస + 1] = 0;
                }
            }
            transpose_nibbles_గమ్యం_tile(tile_dst, tile_quants);
        });
}

స్థిర శూన్య repack_tiled_q4_1(శూన్య * దత్తాంశం, స్థిరరాశి ggml_tensor * t, size_t స్థానభ్రంశం, size_t size) {
    unpack_from_tiled_grid_32<ఖండం_q4_1, 1>(దత్తాంశం, t, స్థానభ్రంశం, size, HTP_TENSOR_TILE_కొలత_Q4_1,
        [](ఖండం_q4_1 * dst_ముక్క, స్థిరరాశి uint8_t * tile_src, int64_t ne0, int64_t ప్రారంభం_అడ్డువరుస, int64_t చివర_అడ్డువరుస, int64_t అడ్డువరుస_stride, పూర్ణాంకం ct, పూర్ణాంకం kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_గమ్యం_nibbles(tile_quants, tile_src);
            స్థిరరాశి ggml_half * scale_src = (స్థిరరాశి ggml_half *) (tile_src + 512);
            కొరకు (పూర్ణాంకం అడ్డువరుస = 0; అడ్డువరుస < 32; అడ్డువరుస++) {
                స్థిరరాశి int64_t r = ct * 32 + అడ్డువరుస;
                ఒకవేళ (r >= ప్రారంభం_అడ్డువరుస && r < చివర_అడ్డువరుస && kt < ne0 / 32) {
                    ఖండం_q4_1 & b = dst_ముక్క[(r - ప్రారంభం_అడ్డువరుస) * అడ్డువరుస_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[అడ్డువరుస]);
                    b.d = scale_src[2 * అడ్డువరుస + 0];
                    b.m = scale_src[2 * అడ్డువరుస + 1];
                }
            }
        });
}

స్థిర శూన్య repack_mxfp4_tiled(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    repack_గమ్యం_tiled_grid_32<ఖండం_mxfp4, 1>(t, దత్తాంశం, స్థానభ్రంశం, size, HTP_TENSOR_TILE_కొలత_MXFP4,
        [](uint8_t * tile_dst, స్థిరరాశి ఖండం_mxfp4 * src_ముక్క, int64_t ne0, int64_t ne1, int64_t అడ్డువరుస_stride, పూర్ణాంకం ct, పూర్ణాంకం kt) {
            uint8_t * scale_dst = tile_dst + 512;
#ఒకవేళ defined(__ARM_NEON) || defined(__aarch64__)
            ఒకవేళ (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                కొరకు (పూర్ణాంకం row0 = 0; row0 < 32; row0 += 8) {
                    స్థిరరాశి uint8_t * qs[8];
                    కొరకు (పూర్ణాంకం rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_ముక్క[(ct * 32 + row0 + rr) * అడ్డువరుస_stride + kt].qs;
                    }
                    repack_neon_4bit_అడ్డువరుసలు(qs, tile_dst, row0);
                }

                కొరకు (పూర్ణాంకం అడ్డువరుస = 0; అడ్డువరుస < 32; ++అడ్డువరుస) {
                    scale_dst[అడ్డువరుస] = src_ముక్క[(ct * 32 + అడ్డువరుస) * అడ్డువరుస_stride + kt].e;
                }
                తిరిగి;
            }
#అయితే_ముగింపు
            uint8_t tile_quants[32][32];
            కొరకు (పూర్ణాంకం అడ్డువరుస = 0; అడ్డువరుస < 32; అడ్డువరుస++) {
                స్థిరరాశి int64_t r = ct * 32 + అడ్డువరుస;
                ఒకవేళ (r < ne1 && kt < ne0 / 32) {
                    స్థిరరాశి ఖండం_mxfp4 & b = src_ముక్క[r * అడ్డువరుస_stride + kt];
                    unpack_nibbles_32(tile_quants[అడ్డువరుస], b.qs);
                    scale_dst[అడ్డువరుస] = b.e;
                } లేకపోతే {
                    memset(tile_quants[అడ్డువరుస], 0, 32);
                    scale_dst[అడ్డువరుస] = 0;
                }
            }
            transpose_nibbles_గమ్యం_tile(tile_dst, tile_quants);
        });
}

స్థిర శూన్య repack_tiled_mxfp4(శూన్య * దత్తాంశం, స్థిరరాశి ggml_tensor * t, size_t స్థానభ్రంశం, size_t size) {
    unpack_from_tiled_grid_32<ఖండం_mxfp4, 1>(దత్తాంశం, t, స్థానభ్రంశం, size, HTP_TENSOR_TILE_కొలత_MXFP4,
        [](ఖండం_mxfp4 * dst_ముక్క, స్థిరరాశి uint8_t * tile_src, int64_t ne0, int64_t ప్రారంభం_అడ్డువరుస, int64_t చివర_అడ్డువరుస, int64_t అడ్డువరుస_stride, పూర్ణాంకం ct, పూర్ణాంకం kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_గమ్యం_nibbles(tile_quants, tile_src);
            స్థిరరాశి uint8_t * scale_src = tile_src + 512;
            కొరకు (పూర్ణాంకం అడ్డువరుస = 0; అడ్డువరుస < 32; అడ్డువరుస++) {
                స్థిరరాశి int64_t r = ct * 32 + అడ్డువరుస;
                ఒకవేళ (r >= ప్రారంభం_అడ్డువరుస && r < చివర_అడ్డువరుస && kt < ne0 / 32) {
                    ఖండం_mxfp4 & b = dst_ముక్క[(r - ప్రారంభం_అడ్డువరుస) * అడ్డువరుస_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[అడ్డువరుస]);
                    b.e = scale_src[అడ్డువరుస];
                }
            }
        });
}

స్థిర శూన్య repack_q8_0_tiled(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    repack_గమ్యం_tiled_grid_32<ఖండం_q8_0, 1>(t, దత్తాంశం, స్థానభ్రంశం, size, HTP_TENSOR_TILE_కొలత_Q8_0,
        [](uint8_t * tile_dst, స్థిరరాశి ఖండం_q8_0 * src_ముక్క, int64_t ne0, int64_t ne1, int64_t అడ్డువరుస_stride, పూర్ణాంకం ct, పూర్ణాంకం kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 1024);
#ఒకవేళ defined(__ARM_NEON) || defined(__aarch64__)
            ఒకవేళ (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                కొరకు (పూర్ణాంకం row0 = 0; row0 < 32; row0 += 8) {
                    స్థిరరాశి uint8_t * qs[8];
                    కొరకు (పూర్ణాంకం rr = 0; rr < 8; ++rr) {
                        qs[rr] = (స్థిరరాశి uint8_t *) src_ముక్క[(ct * 32 + row0 + rr) * అడ్డువరుస_stride + kt].qs;
                    }
                    repack_neon_q8_అడ్డువరుసలు(qs, tile_dst, row0);
                }

                కొరకు (పూర్ణాంకం అడ్డువరుస = 0; అడ్డువరుస < 32; ++అడ్డువరుస) {
                    scale_dst[అడ్డువరుస] = src_ముక్క[(ct * 32 + అడ్డువరుస) * అడ్డువరుస_stride + kt].d;
                }
                తిరిగి;
            }
#అయితే_ముగింపు
            కొరకు (పూర్ణాంకం cp = 0; cp < 16; cp++) {
                స్థిరరాశి పూర్ణాంకం col0 = cp * 2;
                స్థిరరాశి పూర్ణాంకం col1 = col0 + 1;
                కొరకు (పూర్ణాంకం అడ్డువరుస = 0; అడ్డువరుస < 32; అడ్డువరుస++) {
                    స్థిరరాశి int64_t r = ct * 32 + అడ్డువరుస;
                    స్థిరరాశి ఖండం_q8_0 * b = (r < ne1 && kt < ne0 / 32) ? &src_ముక్క[r * అడ్డువరుస_stride + kt] : శూన్యసూచిక;
                    tile_dst[cp * 64 + 2 * అడ్డువరుస + 0] = b ? b->qs[col0] : 0;
                    tile_dst[cp * 64 + 2 * అడ్డువరుస + 1] = b ? b->qs[col1] : 0;
                }
            }
            కొరకు (పూర్ణాంకం అడ్డువరుస = 0; అడ్డువరుస < 32; అడ్డువరుస++) {
                స్థిరరాశి int64_t r = ct * 32 + అడ్డువరుస;
                scale_dst[అడ్డువరుస] = (r < ne1 && kt < ne0 / 32) ? src_ముక్క[r * అడ్డువరుస_stride + kt].d : 0;
            }
        });
}

స్థిర శూన్య repack_tiled_q8_0(శూన్య * దత్తాంశం, స్థిరరాశి ggml_tensor * t, size_t స్థానభ్రంశం, size_t size) {
    unpack_from_tiled_grid_32<ఖండం_q8_0, 1>(దత్తాంశం, t, స్థానభ్రంశం, size, HTP_TENSOR_TILE_కొలత_Q8_0,
        [](ఖండం_q8_0 * dst_ముక్క, స్థిరరాశి uint8_t * tile_src, int64_t ne0, int64_t ప్రారంభం_అడ్డువరుస, int64_t చివర_అడ్డువరుస, int64_t అడ్డువరుస_stride, పూర్ణాంకం ct, పూర్ణాంకం kt) {
            కొరకు (పూర్ణాంకం cp = 0; cp < 16; cp++) {
                స్థిరరాశి పూర్ణాంకం col0 = cp * 2;
                స్థిరరాశి పూర్ణాంకం col1 = col0 + 1;
                కొరకు (పూర్ణాంకం అడ్డువరుస = 0; అడ్డువరుస < 32; అడ్డువరుస++) {
                    స్థిరరాశి int64_t r = ct * 32 + అడ్డువరుస;
                    ఒకవేళ (r >= ప్రారంభం_అడ్డువరుస && r < చివర_అడ్డువరుస && kt < ne0 / 32) {
                        ఖండం_q8_0 & b = dst_ముక్క[(r - ప్రారంభం_అడ్డువరుస) * అడ్డువరుస_stride + kt];
                        b.qs[col0] = tile_src[cp * 64 + 2 * అడ్డువరుస + 0];
                        b.qs[col1] = tile_src[cp * 64 + 2 * అడ్డువరుస + 1];
                    }
                }
            }
            స్థిరరాశి ggml_half * scale_src = (స్థిరరాశి ggml_half *) (tile_src + 1024);
            కొరకు (పూర్ణాంకం అడ్డువరుస = 0; అడ్డువరుస < 32; అడ్డువరుస++) {
                స్థిరరాశి int64_t r = ct * 32 + అడ్డువరుస;
                ఒకవేళ (r >= ప్రారంభం_అడ్డువరుస && r < చివర_అడ్డువరుస && kt < ne0 / 32) {
                    dst_ముక్క[(r - ప్రారంభం_అడ్డువరుస) * అడ్డువరుస_stride + kt].d = scale_src[అడ్డువరుస];
                }
            }
        });
}

// --- Float Types (F16, F32, BF16 -> f16_tiled HMX layout) ---

స్థిర శూన్య repack_q2_0_tiled(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    GGML_ASSERT(స్థానభ్రంశం == 0);
    GGML_ASSERT(t->ne[0] % 64 == 0);
    GGML_ASSERT(size == ggml_nbytes(t));

    స్థిరరాశి ఖండం_q2_0 * src = (స్థిరరాశి ఖండం_q2_0 *) దత్తాంశం;
    స్థిరరాశి int64_t ఖండాలు_చొప్పున_అడ్డువరుస = t->ne[0] / 64;
    స్థిరరాశి int64_t n_నిలువు_tiles = hex_విడత_పైకి(t->ne[1], 32) / 32;
    స్థిరరాశి int64_t n_బరువు_tiles = ఖండాలు_చొప్పున_అడ్డువరుస;
    స్థిరరాశి size_t మాతృక_కొలత = (size_t) n_నిలువు_tiles * n_బరువు_tiles * HTP_TENSOR_TILE_కొలత_Q2_0;

    కొరకు (int64_t ముక్క_2 = 0; ముక్క_2 < t->ne[2] * t->ne[3]; ++ముక్క_2) {
        స్థిరరాశి ఖండం_q2_0 * src_ముక్క = src + ముక్క_2 * t->ne[1] * ఖండాలు_చొప్పున_అడ్డువరుస;
        uint8_t * dst_ముక్క = (uint8_t *) t->దత్తాంశం + ముక్క_2 * మాతృక_కొలత;
        స్థిరరాశి size_t ct_బైట్లు = (size_t) n_బరువు_tiles * HTP_TENSOR_TILE_కొలత_Q2_0;

        repack_parallel_for_ct(మాతృక_కొలత, (పూర్ణాంకం) n_నిలువు_tiles, [&](పూర్ణాంకం ct0, పూర్ణాంకం ct1) {
            memset(dst_ముక్క + (size_t) ct0 * ct_బైట్లు, 0, (size_t) (ct1 - ct0) * ct_బైట్లు);
            స్థిరరాశి int64_t r0 = (int64_t) ct0 * 32;
            స్థిరరాశి int64_t r1 = (std::min)((int64_t) ct1 * 32, t->ne[1]);
            కొరకు (int64_t అడ్డువరుస_సూచీ = r0; అడ్డువరుస_సూచీ < r1; ++అడ్డువరుస_సూచీ) {
                స్థిరరాశి int64_t ct = అడ్డువరుస_సూచీ / 32;
                స్థిరరాశి పూర్ణాంకం అడ్డువరుస = (పూర్ణాంకం) (అడ్డువరుస_సూచీ % 32);
                కొరకు (int64_t kt = 0; kt < n_బరువు_tiles; ++kt) {
                    స్థిరరాశి ఖండం_q2_0 * ఖండం = src_ముక్క + అడ్డువరుస_సూచీ * ఖండాలు_చొప్పున_అడ్డువరుస + kt;
                    uint8_t * tile = dst_ముక్క + (ct * n_బరువు_tiles + kt) * HTP_TENSOR_TILE_కొలత_Q2_0;
#ఒకవేళ defined(__ARM_NEON) || defined(__aarch64__)
                    repack_neon_q2_0_ఖండం(ఖండం->qs, tile, అడ్డువరుస);
#లేకపోతే
                    కొరకు (పూర్ణాంకం సమూహం = 0; సమూహం < 16; ++సమూహం) {
                        కొరకు (పూర్ణాంకం byte = 0; byte < 4; ++byte) {
                            స్థిరరాశి పూర్ణాంకం k = 4 * సమూహం + byte;
                            స్థిరరాశి uint8_t code = (ఖండం->qs[k / 4] >> (2 * (k % 4))) & 3;
                            tile[(సమూహం >> 2) * 128 + 4 * అడ్డువరుస + byte] |= code << (2 * (సమూహం & 3));
                        }
                    }
#అయితే_ముగింపు
                    జ్ఞాపకం_నకలు(tile + 512 + 2 * అడ్డువరుస, &ఖండం->d, పరిమాణం(ఖండం->d));
                }
            }
        });
    }
}

స్థిర శూన్య repack_tiled_q2_0(శూన్య * దత్తాంశం, స్థిరరాశి ggml_tensor * t, size_t స్థానభ్రంశం, size_t size) {
    GGML_ASSERT(t->ne[0] % 64 == 0);
    GGML_ASSERT(స్థానభ్రంశం + size <= ggml_nbytes(t));

    స్థిరరాశి size_t మొదటి_ఖండం = స్థానభ్రంశం / పరిమాణం(ఖండం_q2_0);
    స్థిరరాశి size_t చివర_ఖండం = (స్థానభ్రంశం + size + పరిమాణం(ఖండం_q2_0) - 1) / పరిమాణం(ఖండం_q2_0);
    స్థిరరాశి size_t ఖండాలు_చొప్పున_అడ్డువరుస = t->ne[0] / 64;
    స్థిరరాశి size_t ఖండాలు_చొప్పున_ముక్క = t->ne[1] * ఖండాలు_చొప్పున_అడ్డువరుస;
    స్థిరరాశి size_t n_బరువు_tiles = ఖండాలు_చొప్పున_అడ్డువరుస;
    స్థిరరాశి size_t n_నిలువు_tiles = hex_విడత_పైకి(t->ne[1], 32) / 32;
    స్థిరరాశి size_t మాతృక_కొలత = n_నిలువు_tiles * n_బరువు_tiles * HTP_TENSOR_TILE_కొలత_Q2_0;

    కొరకు (size_t ఖండం_సూచీ = మొదటి_ఖండం; ఖండం_సూచీ < చివర_ఖండం; ++ఖండం_సూచీ) {
        స్థిరరాశి size_t ముక్క_2 = ఖండం_సూచీ / ఖండాలు_చొప్పున_ముక్క;
        స్థిరరాశి size_t లోపల_ముక్క = ఖండం_సూచీ % ఖండాలు_చొప్పున_ముక్క;
        స్థిరరాశి size_t అడ్డువరుస_సూచీ = లోపల_ముక్క / ఖండాలు_చొప్పున_అడ్డువరుస;
        స్థిరరాశి size_t kt = లోపల_ముక్క % ఖండాలు_చొప్పున_అడ్డువరుస;
        స్థిరరాశి size_t ct = అడ్డువరుస_సూచీ / 32;
        స్థిరరాశి size_t అడ్డువరుస = అడ్డువరుస_సూచీ % 32;
        స్థిరరాశి uint8_t * tile = (స్థిరరాశి uint8_t *) t->దత్తాంశం + ముక్క_2 * మాతృక_కొలత +
            (ct * n_బరువు_tiles + kt) * HTP_TENSOR_TILE_కొలత_Q2_0;

        ఖండం_q2_0 ఖండం = {};
        జ్ఞాపకం_నకలు(&ఖండం.d, tile + 512 + 2 * అడ్డువరుస, పరిమాణం(ఖండం.d));
        కొరకు (పూర్ణాంకం సమూహం = 0; సమూహం < 16; ++సమూహం) {
            కొరకు (పూర్ణాంకం byte = 0; byte < 4; ++byte) {
                స్థిరరాశి uint8_t packed = tile[(సమూహం >> 2) * 128 + 4 * అడ్డువరుస + byte];
                స్థిరరాశి uint8_t code = (packed >> (2 * (సమూహం & 3))) & 3;
                ఖండం.qs[సమూహం] |= code << (2 * byte);
            }
        }

        స్థిరరాశి size_t ఖండం_begin = ఖండం_సూచీ * పరిమాణం(ఖండం_q2_0);
        స్థిరరాశి size_t copy_begin = (std::max)(ఖండం_begin, స్థానభ్రంశం);
        స్థిరరాశి size_t copy_చివర = (std::min)(ఖండం_begin + పరిమాణం(ఖండం_q2_0), స్థానభ్రంశం + size);
        జ్ఞాపకం_నకలు((uint8_t *) దత్తాంశం + copy_begin - స్థానభ్రంశం, (స్థిరరాశి uint8_t *) &ఖండం + copy_begin - ఖండం_begin, copy_చివర - copy_begin);
    }
}

మూస <రకం_పేరు SrcT, రకం_పేరు గమ్యం_Fp16_Fn>
స్థిర శూన్య repack_float_గమ్యం_f16_tiled(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size, గమ్యం_Fp16_Fn && గమ్యం_fp16) {
    repack_గమ్యం_tiled_grid_32<SrcT, 32>(t, దత్తాంశం, స్థానభ్రంశం, size, HTP_TENSOR_TILE_కొలత_F16,
        [&](uint8_t * tile_బైట్లు, స్థిరరాశి SrcT * src_ముక్క, int64_t ne0, int64_t ne1, int64_t అడ్డువరుస_stride, పూర్ణాంకం ct, పూర్ణాంకం kt) {
            uint16_t * tile_dst = (uint16_t *) tile_బైట్లు;
#ఒకవేళ defined(__ARM_NEON) || defined(__aarch64__)
            ఒకవేళ (ct * 32 + 32 <= ne1 && kt * 32 + 32 <= ne0) {
                repack_neon_f16_tile(tile_dst, src_ముక్క, అడ్డువరుస_stride, ct, kt);
                తిరిగి;
            }
#అయితే_ముగింపు
            కొరకు (పూర్ణాంకం cp = 0; cp < 16; cp++) {
                స్థిరరాశి int64_t c0 = kt * 32 + cp * 2;
                స్థిరరాశి int64_t c1 = c0 + 1;
                కొరకు (పూర్ణాంకం అడ్డువరుస = 0; అడ్డువరుస < 32; అడ్డువరుస++) {
                    స్థిరరాశి int64_t r = ct * 32 + అడ్డువరుస;
                    స్థిరరాశి SrcT * r_ptr = (r < ne1) ? &src_ముక్క[r * అడ్డువరుస_stride] : శూన్యసూచిక;
                    tile_dst[cp * 64 + అడ్డువరుస * 2 + 0] = (r_ptr && c0 < ne0) ? గమ్యం_fp16(r_ptr[c0]) : 0;
                    tile_dst[cp * 64 + అడ్డువరుస * 2 + 1] = (r_ptr && c1 < ne0) ? గమ్యం_fp16(r_ptr[c1]) : 0;
                }
            }
        });
}

మూస <రకం_పేరు DstT, రకం_పేరు FromFp16Fn>
స్థిర శూన్య unpack_f16_tiled_గమ్యం_float(శూన్య * దత్తాంశం, స్థిరరాశి ggml_tensor * t, size_t స్థానభ్రంశం, size_t size, FromFp16Fn && from_fp16) {
    unpack_from_tiled_grid_32<DstT, 32>(దత్తాంశం, t, స్థానభ్రంశం, size, HTP_TENSOR_TILE_కొలత_F16,
        [&](DstT * dst_ముక్క, స్థిరరాశి uint8_t * tile_బైట్లు, int64_t ne0, int64_t ప్రారంభం_అడ్డువరుస, int64_t చివర_అడ్డువరుస, int64_t అడ్డువరుస_stride, పూర్ణాంకం ct, పూర్ణాంకం kt) {
            స్థిరరాశి uint16_t * tile_src = (స్థిరరాశి uint16_t *) tile_బైట్లు;
            కొరకు (పూర్ణాంకం cp = 0; cp < 16; cp++) {
                స్థిరరాశి int64_t c0 = kt * 32 + cp * 2;
                స్థిరరాశి int64_t c1 = c0 + 1;
                కొరకు (పూర్ణాంకం అడ్డువరుస = 0; అడ్డువరుస < 32; అడ్డువరుస++) {
                    స్థిరరాశి int64_t r = ct * 32 + అడ్డువరుస;
                    ఒకవేళ (r >= ప్రారంభం_అడ్డువరుస && r < చివర_అడ్డువరుస) {
                        DstT * r_ptr = dst_ముక్క + (r - ప్రారంభం_అడ్డువరుస) * అడ్డువరుస_stride;
                        ఒకవేళ (c0 < ne0) r_ptr[c0] = from_fp16(tile_src[cp * 64 + అడ్డువరుస * 2 + 0]);
                        ఒకవేళ (c1 < ne0) r_ptr[c1] = from_fp16(tile_src[cp * 64 + అడ్డువరుస * 2 + 1]);
                    }
                }
            }
        });
}

// --- Generic K-Quant (super-block 256) grid walkers ---

మూస <రకం_పేరు ఖండం_T, రకం_పేరు PackKtFn>
స్థిర శూన్య repack_kquant_గమ్యం_tiled_grid(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size,
                                        size_t tile_కొలత, PackKtFn && pack_kt) {
    GGML_ASSERT(స్థానభ్రంశం == 0);
    GGML_వాడని(size);

    స్థిరరాశి ఖండం_T * src_మాతృక = (స్థిరరాశి ఖండం_T *) దత్తాంశం;
    స్థిరరాశి int64_t ne0 = t->ne[0];
    స్థిరరాశి int64_t ne1 = t->ne[1];
    స్థిరరాశి int64_t ne2 = t->ne[2];
    స్థిరరాశి int64_t ne3 = t->ne[3];
    GGML_ASSERT(ne0 % QK_K == 0);

    స్థిరరాశి పూర్ణాంకం n_నిలువు_tiles    = hex_విడత_పైకి(ne1, 32) / 32;
    స్థిరరాశి పూర్ణాంకం n_k_tiles      = hex_విడత_పైకి(ne0, 32) / 32;
    స్థిరరాశి size_t మాతృక_కొలత = (size_t) n_నిలువు_tiles * n_k_tiles * tile_కొలత;
    స్థిరరాశి int64_t sb_చొప్పున_అడ్డువరుస = ne0 / QK_K;

    కొరకు (int64_t ముక్క_idx = 0; ముక్క_idx < ne2 * ne3; ముక్క_idx++) {
        స్థిరరాశి ఖండం_T * src_ముక్క = src_మాతృక + ముక్క_idx * (ne1 * sb_చొప్పున_అడ్డువరుస);
        uint8_t * మాతృక_dst     = (uint8_t *) t->దత్తాంశం + ముక్క_idx * మాతృక_కొలత;
        స్థిరరాశి size_t ct_బైట్లు    = (size_t) n_k_tiles * tile_కొలత;

        repack_parallel_for_ct(మాతృక_కొలత, n_నిలువు_tiles, [&](పూర్ణాంకం ct0, పూర్ణాంకం ct1) {
            // Padding rows and OR-ed bit planes require zeroed tiles
            memset(మాతృక_dst + (size_t) ct0 * ct_బైట్లు, 0, (size_t) (ct1 - ct0) * ct_బైట్లు);

            స్థిరరాశి int64_t r0 = (int64_t) ct0 * 32;
            స్థిరరాశి int64_t r1 = (std::min)((int64_t) ct1 * 32, ne1);
            కొరకు (int64_t r = r0; r < r1; r++) {
                స్థిరరాశి పూర్ణాంకం ct  = (పూర్ణాంకం) (r / 32);
                స్థిరరాశి పూర్ణాంకం అడ్డువరుస = (పూర్ణాంకం) (r % 32);
                స్థిరరాశి ఖండం_T * src_అడ్డువరుస = src_ముక్క + r * sb_చొప్పున_అడ్డువరుస;

                కొరకు (పూర్ణాంకం kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile = మాతృక_dst + ((size_t) ct * n_k_tiles + kt) * tile_కొలత;
                    pack_kt(tile, &src_అడ్డువరుస[kt / 8], అడ్డువరుస, kt % 8);
                }
            }
        });
    }
}

మూస <రకం_పేరు ఖండం_T, రకం_పేరు UnpackSbFn>
స్థిర శూన్య unpack_kquant_from_tiled_grid(శూన్య * దత్తాంశం, స్థిరరాశి ggml_tensor * t, size_t స్థానభ్రంశం, size_t size,
                                          size_t tile_కొలత, UnpackSbFn && unpack_sb) {
    uint8_t * dst_బైట్లు = (uint8_t *) దత్తాంశం;
    స్థిరరాశి int64_t ne0 = t->ne[0];
    స్థిరరాశి int64_t ne1 = t->ne[1];
    స్థిరరాశి int64_t ne2 = t->ne[2];
    స్థిరరాశి int64_t ne3 = t->ne[3];
    GGML_ASSERT(ne0 % QK_K == 0);

    స్థిరరాశి పూర్ణాంకం n_నిలువు_tiles    = hex_విడత_పైకి(ne1, 32) / 32;
    స్థిరరాశి పూర్ణాంకం n_k_tiles      = hex_విడత_పైకి(ne0, 32) / 32;
    స్థిరరాశి size_t మాతృక_కొలత = (size_t) n_నిలువు_tiles * n_k_tiles * tile_కొలత;
    స్థిరరాశి int64_t sb_చొప్పున_అడ్డువరుస = ne0 / QK_K;

    స్థిరరాశి size_t అడ్డువరుస_కొలత_బైట్లు = ggml_అడ్డువరుస_కొలత(t->type, ne0);
    స్థిరరాశి size_t ముక్క_కొలత     = (size_t) ne1 * అడ్డువరుస_కొలత_బైట్లు;
    స్థిరరాశి int64_t ప్రారంభం_ముక్క   = స్థానభ్రంశం / ముక్క_కొలత;
    int64_t చివర_ముక్క           = (స్థానభ్రంశం + size + ముక్క_కొలత - 1) / ముక్క_కొలత;
    ఒకవేళ (చివర_ముక్క > ne2 * ne3) {
        చివర_ముక్క = ne2 * ne3;
    }

    కొరకు (int64_t ముక్క_idx = ప్రారంభం_ముక్క; ముక్క_idx < చివర_ముక్క; ముక్క_idx++) {
        స్థిరరాశి size_t cur_ప్రారంభం_byte     = (std::max)(స్థానభ్రంశం, (size_t) ముక్క_idx * ముక్క_కొలత);
        స్థిరరాశి size_t cur_చివర_byte       = (std::min)(స్థానభ్రంశం + size, (size_t) (ముక్క_idx + 1) * ముక్క_కొలత);
        స్థిరరాశి size_t ముక్క_స్థానభ్రంశం_ప్రారంభం = cur_ప్రారంభం_byte - (size_t) ముక్క_idx * ముక్క_కొలత;
        స్థిరరాశి size_t ముక్క_స్థానభ్రంశం_చివర   = cur_చివర_byte - (size_t) ముక్క_idx * ముక్క_కొలత;

        స్థిరరాశి int64_t ప్రారంభం_అడ్డువరుస = ముక్క_స్థానభ్రంశం_ప్రారంభం / అడ్డువరుస_కొలత_బైట్లు;
        స్థిరరాశి int64_t చివర_అడ్డువరుస   = (std::min)((int64_t) ((ముక్క_స్థానభ్రంశం_చివర + అడ్డువరుస_కొలత_బైట్లు - 1) / అడ్డువరుస_కొలత_బైట్లు), ne1);

        ఖండం_T * dst_ముక్క         = (ఖండం_T *) (dst_బైట్లు + (cur_ప్రారంభం_byte - స్థానభ్రంశం));
        స్థిరరాశి uint8_t * మాతృక_src = (స్థిరరాశి uint8_t *) t->దత్తాంశం + ముక్క_idx * మాతృక_కొలత;

        కొరకు (int64_t r = ప్రారంభం_అడ్డువరుస; r < చివర_అడ్డువరుస; r++) {
            స్థిరరాశి పూర్ణాంకం ct  = (పూర్ణాంకం) (r / 32);
            స్థిరరాశి పూర్ణాంకం అడ్డువరుస = (పూర్ణాంకం) (r % 32);
            ఖండం_T * dst_అడ్డువరుస = dst_ముక్క + (r - ప్రారంభం_అడ్డువరుస) * sb_చొప్పున_అడ్డువరుస;

            కొరకు (int64_t sb = 0; sb < sb_చొప్పున_అడ్డువరుస; sb++) {
                ఖండం_T * b = &dst_అడ్డువరుస[sb];
                memset(b, 0, పరిమాణం(ఖండం_T));
                స్థిరరాశి uint8_t * sb_tiles = మాతృక_src + ((size_t) ct * n_k_tiles + sb * 8) * tile_కొలత;
                unpack_sb(b, sb_tiles, tile_కొలత, అడ్డువరుస);
            }
        }
    }
}

// --- K-Quant Helpers & Per-Type Implementations (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) ---

#ఒకవేళ !defined(__ARM_NEON) && !defined(__aarch64__)
// unsigned 6-bit value (0..63) of element e of a Q6_K block, same bit layout as dequantize_row_q6_K
స్థిర అంతఃవరుస uint8_t q6_K_get_quant(స్థిరరాశి ఖండం_q6_K * b, పూర్ణాంకం e) {
    స్థిరరాశి పూర్ణాంకం c = e / 128;
    స్థిరరాశి పూర్ణాంకం w = e % 128;
    స్థిరరాశి పూర్ణాంకం g = w / 32;
    స్థిరరాశి పూర్ణాంకం l = w % 32;
    స్థిరరాశి uint8_t * ql = b->ql + c * 64;
    స్థిరరాశి uint8_t * qh = b->qh + c * 32;
    uint8_t lo, hi;
    ఎంపిక (g) {
        స్థితి 0:  lo = ql[l]      & 0xF; hi = (qh[l] >> 0) & 3; ఆపు;
        స్థితి 1:  lo = ql[l + 32] & 0xF; hi = (qh[l] >> 2) & 3; ఆపు;
        స్థితి 2:  lo = ql[l]      >> 4;  hi = (qh[l] >> 4) & 3; ఆపు;
        అప్రమేయ: lo = ql[l + 32] >> 4;  hi = (qh[l] >> 6) & 3; ఆపు;
    }
    తిరిగి (uint8_t) (lo | (hi << 4));
}
#అయితే_ముగింపు

స్థిర శూన్య repack_q6_K_tiled(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    repack_kquant_గమ్యం_tiled_grid<ఖండం_q6_K>(t, దత్తాంశం, స్థానభ్రంశం, size, HTP_TENSOR_TILE_కొలత_Q6_K,
        [](uint8_t * tile, స్థిరరాశి ఖండం_q6_K * b, పూర్ణాంకం అడ్డువరుస, పూర్ణాంకం kt_local) {
            స్థిరరాశి దశాంశం d     = GGML_FP16_గమ్యం_FP32(b->d);
            uint8_t * lo_pl   = tile;
            uint8_t * hi_pl   = tile + 512;
            ggml_half * sc_pl = (ggml_half *) (tile + 768);

#ఒకవేళ defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q6_k(b, kt_local, lo_pl, hi_pl, అడ్డువరుస);
#లేకపోతే
            కొరకు (పూర్ణాంకం lk = 0; lk < 32; lk++) {
                స్థిరరాశి uint8_t q6 = q6_K_get_quant(b, kt_local * 32 + lk);
                స్థిరరాశి పూర్ణాంకం g   = lk >> 2;
                స్థిరరాశి పూర్ణాంకం pos = అడ్డువరుస * 4 + (lk & 3);
                lo_pl[(g >> 1) * 128 + pos] |= (uint8_t) ((q6 & 0xF) << ((g & 1) * 4));
                hi_pl[(g >> 2) * 128 + pos] |= (uint8_t) ((q6 >> 4) << ((g & 3) * 2));
            }
#అయితే_ముగింపు
            కొరకు (పూర్ణాంకం sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + అడ్డువరుస] = GGML_FP32_గమ్యం_FP16(d * (దశాంశం) b->scales[kt_local * 2 + sub]);
            }
        });
}

స్థిర శూన్య repack_tiled_q6_K(శూన్య * దత్తాంశం, స్థిరరాశి ggml_tensor * t, size_t స్థానభ్రంశం, size_t size) {
    unpack_kquant_from_tiled_grid<ఖండం_q6_K>(దత్తాంశం, t, స్థానభ్రంశం, size, HTP_TENSOR_TILE_కొలత_Q6_K,
        [](ఖండం_q6_K * b, స్థిరరాశి uint8_t * sb_tiles, size_t tile_కొలత, పూర్ణాంకం అడ్డువరుస) {
            దశాంశం sub_scales[16];
            కొరకు (పూర్ణాంకం kt_local = 0; kt_local < 8; kt_local++) {
                స్థిరరాశి uint8_t *   tile  = sb_tiles + kt_local * tile_కొలత;
                స్థిరరాశి uint8_t *   lo_pl = tile;
                స్థిరరాశి uint8_t *   hi_pl = tile + 512;
                స్థిరరాశి ggml_half * sc_pl = (స్థిరరాశి ggml_half *) (tile + 768);

                స్థిరరాశి పూర్ణాంకం c  = kt_local / 4;
                స్థిరరాశి పూర్ణాంకం g  = kt_local % 4;
                uint8_t * ql = b->ql + c * 64;
                uint8_t * qh = b->qh + c * 32;

                కొరకు (పూర్ణాంకం lk = 0; lk < 32; lk++) {
                    స్థిరరాశి పూర్ణాంకం g_tile = lk >> 2;
                    స్థిరరాశి పూర్ణాంకం pos    = అడ్డువరుస * 4 + (lk & 3);
                    స్థిరరాశి uint8_t lo = (lo_pl[(g_tile >> 1) * 128 + pos] >> ((g_tile & 1) * 4)) & 0xF;
                    స్థిరరాశి uint8_t hi = (hi_pl[(g_tile >> 2) * 128 + pos] >> ((g_tile & 3) * 2)) & 3;

                    ఎంపిక (g) {
                        స్థితి 0:  ql[lk]      |= lo;        qh[lk] |= (hi << 0); ఆపు;
                        స్థితి 1:  ql[lk + 32] |= lo;        qh[lk] |= (hi << 2); ఆపు;
                        స్థితి 2:  ql[lk]      |= (lo << 4); qh[lk] |= (hi << 4); ఆపు;
                        అప్రమేయ: ql[lk + 32] |= (lo << 4); qh[lk] |= (hi << 6); ఆపు;
                    }
                }

                కొరకు (పూర్ణాంకం sub = 0; sub < 2; sub++) {
                    sub_scales[kt_local * 2 + sub] = GGML_FP16_గమ్యం_FP32(sc_pl[sub * 32 + అడ్డువరుస]);
                }
            }

            దశాంశం max_abs_scale = 0.0f;
            కొరకు (పూర్ణాంకం s = 0; s < 16; s++) {
                max_abs_scale = (std::max)(max_abs_scale, fabsf(sub_scales[s]));
            }

            ఒకవేళ (max_abs_scale == 0.0f) {
                b->d = GGML_FP32_గమ్యం_FP16(0.0f);
            } లేకపోతే {
                b->d = GGML_FP32_గమ్యం_FP16(max_abs_scale / 127.0f);
                స్థిరరాశి దశాంశం d_actual = GGML_FP16_గమ్యం_FP32(b->d);
                స్థిరరాశి దశాంశం inv_d    = (d_actual != 0.0f) ? (1.0f / d_actual) : 0.0f;
                కొరకు (పూర్ణాంకం s = 0; s < 16; s++) {
                    స్థిరరాశి పూర్ణాంకం sc = (పూర్ణాంకం) roundf(sub_scales[s] * inv_d);
                    b->scales[s] = (int8_t) (std::max)(-128, (std::min)(127, sc));
                }
            }
        });
}

#ఒకవేళ !defined(__ARM_NEON) && !defined(__aarch64__)
// low 2 bits (0..3) of element e of a Q2_K or Q3_K block, same bit layout as dequantize_row_q2_K / q3_K
స్థిర అంతఃవరుస uint8_t q2_3_K_get_low2(స్థిరరాశి uint8_t * qs, పూర్ణాంకం e) {
    స్థిరరాశి పూర్ణాంకం c = e / 128;
    స్థిరరాశి పూర్ణాంకం j = (e % 128) / 32;
    స్థిరరాశి పూర్ణాంకం l = e % 32;
    తిరిగి (qs[c * 32 + l] >> (2 * j)) & 3;
}

// hmask bit of element e of a Q3_K block, same bit layout as dequantize_row_q3_K
స్థిర అంతఃవరుస తార్కిక q3_K_get_hbit(స్థిరరాశి ఖండం_q3_K * b, పూర్ణాంకం e) {
    స్థిరరాశి పూర్ణాంకం c = e / 128;
    స్థిరరాశి పూర్ణాంకం j = (e % 128) / 32;
    స్థిరరాశి పూర్ణాంకం l = e % 32;
    తిరిగి (b->hmask[l] >> (c * 4 + j)) & 1;
}
#అయితే_ముగింపు

// signed 6-bit scale j (-32..31) of a Q3_K block, same packing as quantize_row_q3_K_ref
స్థిర అంతఃవరుస పూర్ణాంకం q3_K_get_scale(స్థిరరాశి uint8_t * scales, పూర్ణాంకం j) {
    స్థిరరాశి పూర్ణాంకం lo = (j < 8) ? (scales[j] & 0xF) : (scales[j - 8] >> 4);
    స్థిరరాశి పూర్ణాంకం hi = (scales[8 + j % 4] >> (2 * (j / 4))) & 3;
    తిరిగి (lo | (hi << 4)) - 32;
}

// read-back: find fp16 d and l[j] in [lmin, lmax] with fp16(d * l[j]) == prod[j] for all j, false if none
స్థిర తార్కిక hexagon_recover_k_scales(స్థిరరాశి ggml_half * prod, పూర్ణాంకం n, పూర్ణాంకం lmin, పూర్ణాంకం lmax, ggml_half * d_బహిర్గతం, పూర్ణాంకం * l_బహిర్గతం) {
    పూర్ణాంకం jmax = 0;
    కొరకు (పూర్ణాంకం j = 1; j < n; j++) {
        ఒకవేళ (fabsf(GGML_FP16_గమ్యం_FP32(prod[j])) > fabsf(GGML_FP16_గమ్యం_FP32(prod[jmax]))) {
            jmax = j;
        }
    }
    స్థిరరాశి దశాంశం pmax = GGML_FP16_గమ్యం_FP32(prod[jmax]);
    ఒకవేళ (pmax == 0.0f) {
        *d_బహిర్గతం = GGML_FP32_గమ్యం_FP16(0.0f);
        కొరకు (పూర్ణాంకం j = 0; j < n; j++) {
            l_బహిర్గతం[j] = 0;
        }
        తిరిగి నిజం;
    }

    // the quantizers put the largest scale at or near the range end, so try large |l| first
    స్థిరరాశి పూర్ణాంకం lext = (std::max)(-lmin, lmax);
    కొరకు (పూర్ణాంకం a = lext; a >= 1; a--) {
        కొరకు (పూర్ణాంకం sign : { 1, -1 }) {
            స్థిరరాశి పూర్ణాంకం lj = sign * a;
            ఒకవేళ (lj < lmin || lj > lmax) {
                కొనసాగు;
            }
            స్థిరరాశి ggml_half d0 = GGML_FP32_గమ్యం_FP16(pmax / (దశాంశం) lj);
            కొరకు (పూర్ణాంకం ulp : { 0, -1, 1 }) {
                ggml_half d = d0;
                uint16_t బిట్లు;
                జ్ఞాపకం_నకలు(&బిట్లు, &d, పరిమాణం(బిట్లు));
                బిట్లు = (uint16_t) (బిట్లు + ulp);
                జ్ఞాపకం_నకలు(&d, &బిట్లు, పరిమాణం(బిట్లు));

                స్థిరరాశి దశాంశం df = GGML_FP16_గమ్యం_FP32(d);
                ఒకవేళ (!std::isfinite(df) || df == 0.0f) {
                    కొనసాగు;
                }
                తార్కిక ok = నిజం;
                కొరకు (పూర్ణాంకం j = 0; j < n && ok; j++) {
                    స్థిరరాశి పూర్ణాంకం l = (పూర్ణాంకం) roundf(GGML_FP16_గమ్యం_FP32(prod[j]) / df);
                    స్థిరరాశి ggml_half p = GGML_FP32_గమ్యం_FP16(df * (దశాంశం) l);
                    ok = l >= lmin && l <= lmax && memcmp(&p, &prod[j], పరిమాణం(p)) == 0;
                    l_బహిర్గతం[j] = l;
                }
                ఒకవేళ (ok) {
                    *d_బహిర్గతం = d;
                    తిరిగి నిజం;
                }
            }
        }
    }
    తిరిగి అబద్ధం;
}

స్థిర శూన్య repack_q3_K_tiled(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    repack_kquant_గమ్యం_tiled_grid<ఖండం_q3_K>(t, దత్తాంశం, స్థానభ్రంశం, size, HTP_TENSOR_TILE_కొలత_Q3_K,
        [](uint8_t * tile, స్థిరరాశి ఖండం_q3_K * b, పూర్ణాంకం అడ్డువరుస, పూర్ణాంకం kt_local) {
            స్థిరరాశి దశాంశం d     = GGML_FP16_గమ్యం_FP32(b->d);
            uint8_t * lo_pl   = tile;
            uint8_t * neg_pl  = tile + 256;
            ggml_half * sc_pl = (ggml_half *) (tile + 384);

#ఒకవేళ defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q2_3_k_low2(b->qs, kt_local, lo_pl, అడ్డువరుస);
            repack_neon_q3_k_hmask(b->hmask, kt_local, neg_pl, అడ్డువరుస);
#లేకపోతే
            కొరకు (పూర్ణాంకం lk = 0; lk < 32; lk++) {
                స్థిరరాశి పూర్ణాంకం e   = kt_local * 32 + lk;
                స్థిరరాశి పూర్ణాంకం g   = lk >> 2;
                స్థిరరాశి పూర్ణాంకం pos = అడ్డువరుస * 4 + (lk & 3);
                lo_pl[(g >> 2) * 128 + pos] |= (uint8_t) (q2_3_K_get_low2(b->qs, e) << ((g & 3) * 2));
                ఒకవేళ (!q3_K_get_hbit(b, e)) {
                    neg_pl[pos] |= (uint8_t) (1 << g);
                }
            }
#అయితే_ముగింపు
            కొరకు (పూర్ణాంకం sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + అడ్డువరుస] = GGML_FP32_గమ్యం_FP16(d * (దశాంశం) q3_K_get_scale(b->scales, kt_local * 2 + sub));
            }
        });
}

స్థిర శూన్య repack_tiled_q3_K(శూన్య * దత్తాంశం, స్థిరరాశి ggml_tensor * t, size_t స్థానభ్రంశం, size_t size) {
    unpack_kquant_from_tiled_grid<ఖండం_q3_K>(దత్తాంశం, t, స్థానభ్రంశం, size, HTP_TENSOR_TILE_కొలత_Q3_K,
        [](ఖండం_q3_K * b, స్థిరరాశి uint8_t * sb_tiles, size_t tile_కొలత, పూర్ణాంకం అడ్డువరుస) {
            ggml_half sub_scales[16];
            కొరకు (పూర్ణాంకం kt_local = 0; kt_local < 8; kt_local++) {
                స్థిరరాశి uint8_t *   tile   = sb_tiles + kt_local * tile_కొలత;
                స్థిరరాశి uint8_t *   lo_pl  = tile;
                స్థిరరాశి uint8_t *   neg_pl = tile + 256;
                స్థిరరాశి ggml_half * sc_pl  = (స్థిరరాశి ggml_half *) (tile + 384);

                కొరకు (పూర్ణాంకం lk = 0; lk < 32; lk++) {
                    స్థిరరాశి పూర్ణాంకం e   = kt_local * 32 + lk;
                    స్థిరరాశి పూర్ణాంకం g   = lk >> 2;
                    స్థిరరాశి పూర్ణాంకం pos = అడ్డువరుస * 4 + (lk & 3);
                    స్థిరరాశి uint8_t lo = (lo_pl[(g >> 2) * 128 + pos] >> ((g & 3) * 2)) & 3;

                    స్థిరరాశి పూర్ణాంకం c = e / 128;
                    స్థిరరాశి పూర్ణాంకం j = (e % 128) / 32;
                    స్థిరరాశి పూర్ణాంకం l = e % 32;
                    b->qs[c * 32 + l] |= (uint8_t) (lo << (2 * j));
                    ఒకవేళ (!((neg_pl[pos] >> g) & 1)) {
                        b->hmask[l] |= (uint8_t) (1 << (c * 4 + j));
                    }
                }

                కొరకు (పూర్ణాంకం sub = 0; sub < 2; sub++) {
                    sub_scales[kt_local * 2 + sub] = sc_pl[sub * 32 + అడ్డువరుస];
                }
            }

            పూర్ణాంకం ls[16];
            ఒకవేళ (!hexagon_recover_k_scales(sub_scales, 16, -32, 31, &b->d, ls)) {
                దశాంశం max_scale = 0.0f;
                కొరకు (పూర్ణాంకం s = 0; s < 16; s++) {
                    ఒకవేళ (fabsf(GGML_FP16_గమ్యం_FP32(sub_scales[s])) > fabsf(max_scale)) {
                        max_scale = GGML_FP16_గమ్యం_FP32(sub_scales[s]);
                    }
                }
                b->d = GGML_FP32_గమ్యం_FP16(-max_scale / 32.0f);
                స్థిరరాశి దశాంశం d_actual = GGML_FP16_గమ్యం_FP32(b->d);
                స్థిరరాశి దశాంశం inv_d    = (d_actual != 0.0f) ? (1.0f / d_actual) : 0.0f;
                కొరకు (పూర్ణాంకం s = 0; s < 16; s++) {
                    ls[s] = (std::max)(-32, (std::min)(31, (పూర్ణాంకం) roundf(GGML_FP16_గమ్యం_FP32(sub_scales[s]) * inv_d)));
                }
            }

            కొరకు (పూర్ణాంకం s = 0; s < 16; s++) {
                స్థిరరాశి పూర్ణాంకం l = ls[s] + 32;
                ఒకవేళ (s < 8) {
                    b->scales[s] = l & 0xF;
                } లేకపోతే {
                    b->scales[s - 8] |= (uint8_t) ((l & 0xF) << 4);
                }
                b->scales[s % 4 + 8] |= (uint8_t) ((l >> 4) << (2 * (s / 4)));
            }
        });
}

స్థిర శూన్య repack_q2_K_tiled(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    repack_kquant_గమ్యం_tiled_grid<ఖండం_q2_K>(t, దత్తాంశం, స్థానభ్రంశం, size, HTP_TENSOR_TILE_కొలత_Q2_K,
        [](uint8_t * tile, స్థిరరాశి ఖండం_q2_K * b, పూర్ణాంకం అడ్డువరుస, పూర్ణాంకం kt_local) {
            uint8_t * lo_pl   = tile;
            uint8_t * sc_pl   = tile + 256;
            ggml_half * dd_pl = (ggml_half *) (tile + 320);

#ఒకవేళ defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q2_3_k_low2(b->qs, kt_local, lo_pl, అడ్డువరుస);
#లేకపోతే
            కొరకు (పూర్ణాంకం lk = 0; lk < 32; lk++) {
                స్థిరరాశి పూర్ణాంకం g   = lk >> 2;
                స్థిరరాశి పూర్ణాంకం pos = అడ్డువరుస * 4 + (lk & 3);
                lo_pl[(g >> 2) * 128 + pos] |= (uint8_t) (q2_3_K_get_low2(b->qs, kt_local * 32 + lk) << ((g & 3) * 2));
            }
#అయితే_ముగింపు
            కొరకు (పూర్ణాంకం sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + అడ్డువరుస] = b->scales[kt_local * 2 + sub];
            }
            ఒకవేళ (అడ్డువరుస / 8 == kt_local % 4) {
                dd_pl[అడ్డువరుస % 8] = kt_local < 4 ? b->d : (ggml_half) (b->dmin ^ 0x8000);
            }
        });
}

స్థిర శూన్య repack_tiled_q2_K(శూన్య * దత్తాంశం, స్థిరరాశి ggml_tensor * t, size_t స్థానభ్రంశం, size_t size) {
    unpack_kquant_from_tiled_grid<ఖండం_q2_K>(దత్తాంశం, t, స్థానభ్రంశం, size, HTP_TENSOR_TILE_కొలత_Q2_K,
        [](ఖండం_q2_K * b, స్థిరరాశి uint8_t * sb_tiles, size_t tile_కొలత, పూర్ణాంకం అడ్డువరుస) {
            కొరకు (పూర్ణాంకం kt_local = 0; kt_local < 8; kt_local++) {
                స్థిరరాశి uint8_t *   tile  = sb_tiles + kt_local * tile_కొలత;
                స్థిరరాశి uint8_t *   lo_pl = tile;
                స్థిరరాశి uint8_t *   sc_pl = tile + 256;
                స్థిరరాశి ggml_half * dd_pl = (స్థిరరాశి ggml_half *) (tile + 320);

                కొరకు (పూర్ణాంకం lk = 0; lk < 32; lk++) {
                    స్థిరరాశి పూర్ణాంకం e   = kt_local * 32 + lk;
                    స్థిరరాశి పూర్ణాంకం g   = lk >> 2;
                    స్థిరరాశి పూర్ణాంకం pos = అడ్డువరుస * 4 + (lk & 3);
                    స్థిరరాశి uint8_t lo = (lo_pl[(g >> 2) * 128 + pos] >> ((g & 3) * 2)) & 3;
                    b->qs[(e / 128) * 32 + e % 32] |= (uint8_t) (lo << (2 * ((e % 128) / 32)));
                }

                కొరకు (పూర్ణాంకం sub = 0; sub < 2; sub++) {
                    b->scales[kt_local * 2 + sub] = sc_pl[sub * 32 + అడ్డువరుస];
                }
                ఒకవేళ (అడ్డువరుస / 8 == kt_local % 4) {
                    ఒకవేళ (kt_local < 4) {
                        b->d = dd_pl[అడ్డువరుస % 8];
                    } లేకపోతే {
                        b->dmin = (ggml_half) (dd_pl[అడ్డువరుస % 8] ^ 0x8000);
                    }
                }
            }
        });
}

స్థిర అంతఃవరుస శూన్య get_scale_min_k4(పూర్ణాంకం j, స్థిరరాశి uint8_t * q, uint8_t * d, uint8_t * m) {
    ఒకవేళ (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } లేకపోతే {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

స్థిర అంతఃవరుస శూన్య pack_k4_scales_mins(uint8_t * scales, ggml_half * d, ggml_half * dmin,
                                       స్థిరరాశి దశాంశం * sub_scales, స్థిరరాశి దశాంశం * sub_mins) {
    దశాంశం max_scale = 0.0f;
    దశాంశం max_min   = 0.0f;
    కొరకు (పూర్ణాంకం j = 0; j < 8; j++) {
        ఒకవేళ (sub_scales[j] > max_scale) max_scale = sub_scales[j];
        ఒకవేళ (sub_mins[j]   > max_min)   max_min   = sub_mins[j];
    }

    దశాంశం inv_scale = 0.0f;
    ఒకవేళ (max_scale > 0.0f) {
        *d = GGML_FP32_గమ్యం_FP16(max_scale / 63.0f);
        స్థిరరాశి దశాంశం d_actual = GGML_FP16_గమ్యం_FP32(*d);
        inv_scale = (d_actual > 0.0f) ? (1.0f / d_actual) : 0.0f;
    } లేకపోతే {
        *d = GGML_FP32_గమ్యం_FP16(0.0f);
    }

    దశాంశం inv_min = 0.0f;
    ఒకవేళ (max_min > 0.0f) {
        *dmin = GGML_FP32_గమ్యం_FP16(max_min / 63.0f);
        స్థిరరాశి దశాంశం dmin_actual = GGML_FP16_గమ్యం_FP32(*dmin);
        inv_min = (dmin_actual > 0.0f) ? (1.0f / dmin_actual) : 0.0f;
    } లేకపోతే {
        *dmin = GGML_FP32_గమ్యం_FP16(0.0f);
    }

    కొరకు (పూర్ణాంకం j = 0; j < 8; j++) {
        uint8_t ls = (std::min)((uint8_t) 63, (uint8_t) roundf(inv_scale * sub_scales[j]));
        uint8_t lm = (std::min)((uint8_t) 63, (uint8_t) roundf(inv_min * sub_mins[j]));
        ఒకవేళ (j < 4) {
            scales[j]     = ls;
            scales[j + 4] = lm;
        } లేకపోతే {
            scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            scales[j - 4] |= ((ls >> 4) << 6);
            scales[j - 0] |= ((lm >> 4) << 6);
        }
    }
}

మూస <రకం_పేరు ఖండం_T>
స్థిర అంతఃవరుస శూన్య repack_k4_quants(uint8_t * tile_dst, స్థిరరాశి ఖండం_T * b, పూర్ణాంకం అడ్డువరుస, పూర్ణాంకం kt_local,
                                    int64_t sb_చొప్పున_అడ్డువరుస, తార్కిక use_neon) {
#ఒకవేళ defined(__ARM_NEON) || defined(__aarch64__)
    ఒకవేళ (use_neon) {
        ఒకవేళ ((అడ్డువరుస & 7) == 0) {
            స్థిరరాశి uint8_t * qs[8];
            కొరకు (పూర్ణాంకం rr = 0; rr < 8; ++rr) {
                qs[rr] = b[rr * sb_చొప్పున_అడ్డువరుస].qs + (kt_local / 2) * 32;
            }
            repack_neon_k4_అడ్డువరుసలు(qs, tile_dst, అడ్డువరుస, (kt_local & 1) != 0);
        }
        తిరిగి;
    }
#లేకపోతే
    GGML_వాడని(sb_చొప్పున_అడ్డువరుస);
    GGML_వాడని(use_neon);
#అయితే_ముగింపు
    స్థిరరాశి uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
    స్థిరరాశి పూర్ణాంకం జరుపు = (kt_local & 1) ? 4 : 0;

    కొరకు (పూర్ణాంకం cp = 0; cp < 16; cp++) {
        స్థిరరాశి uint8_t q0 = (qs_sub[2 * cp + 0] >> జరుపు) & 0x0F;
        స్థిరరాశి uint8_t q1 = (qs_sub[2 * cp + 1] >> జరుపు) & 0x0F;
        tile_dst[cp * 32 + అడ్డువరుస] = (uint8_t) ((q1 << 4) | q0);
    }
}

స్థిర శూన్య repack_q4_K_tiled(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    స్థిరరాశి int64_t sb_చొప్పున_అడ్డువరుస = t->ne[0] / QK_K;
    స్థిరరాశి తార్కిక    use_neon   = (t->ne[1] % 8 == 0);
    repack_kquant_గమ్యం_tiled_grid<ఖండం_q4_K>(t, దత్తాంశం, స్థానభ్రంశం, size, HTP_TENSOR_TILE_కొలత_Q4_K,
        [sb_చొప్పున_అడ్డువరుస, use_neon](uint8_t * tile_dst, స్థిరరాశి ఖండం_q4_K * b, పూర్ణాంకం అడ్డువరుస, పూర్ణాంకం kt_local) {
            స్థిరరాశి దశాంశం d    = GGML_FP16_గమ్యం_FP32(b->d);
            స్థిరరాశి దశాంశం dmin = GGML_FP16_గమ్యం_FP32(b->dmin);

            uint8_t sc, m;
            get_scale_min_k4(kt_local, b->scales, &sc, &m);

            repack_k4_quants(tile_dst, b, అడ్డువరుస, kt_local, sb_చొప్పున_అడ్డువరుస, use_neon);

            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
            scale_dst[2 * అడ్డువరుస + 0] = GGML_FP32_గమ్యం_FP16(d * (దశాంశం) sc);
            scale_dst[2 * అడ్డువరుస + 1] = GGML_FP32_గమ్యం_FP16(-dmin * (దశాంశం) m);
        });
}

స్థిర శూన్య repack_tiled_q4_K(శూన్య * దత్తాంశం, స్థిరరాశి ggml_tensor * t, size_t స్థానభ్రంశం, size_t size) {
    unpack_kquant_from_tiled_grid<ఖండం_q4_K>(దత్తాంశం, t, స్థానభ్రంశం, size, HTP_TENSOR_TILE_కొలత_Q4_K,
        [](ఖండం_q4_K * b, స్థిరరాశి uint8_t * sb_tiles, size_t tile_కొలత, పూర్ణాంకం అడ్డువరుస) {
            దశాంశం sub_scales[8];
            దశాంశం sub_mins[8];

            కొరకు (పూర్ణాంకం kt_local = 0; kt_local < 8; kt_local++) {
                స్థిరరాశి uint8_t *   tile_src  = sb_tiles + kt_local * tile_కొలత;
                స్థిరరాశి ggml_half * scale_src = (స్థిరరాశి ggml_half *) (tile_src + 512);

                uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
                స్థిరరాశి పూర్ణాంకం జరుపు  = (kt_local & 1) ? 4 : 0;

                కొరకు (పూర్ణాంకం cp = 0; cp < 16; cp++) {
                    స్థిరరాశి uint8_t val = tile_src[cp * 32 + అడ్డువరుస];
                    qs_sub[2 * cp + 0] |= (uint8_t) ((val & 0x0F) << జరుపు);
                    qs_sub[2 * cp + 1] |= (uint8_t) ((val >> 4)   << జరుపు);
                }

                స్థిరరాశి దశాంశం D = GGML_FP16_గమ్యం_FP32(scale_src[2 * అడ్డువరుస + 0]);
                స్థిరరాశి దశాంశం M = GGML_FP16_గమ్యం_FP32(scale_src[2 * అడ్డువరుస + 1]);
                sub_scales[kt_local] = (D > 0.0f) ? D : 0.0f;
                sub_mins[kt_local]   = (-M > 0.0f) ? -M : 0.0f;
            }

            pack_k4_scales_mins(b->scales, &b->d, &b->dmin, sub_scales, sub_mins);
        });
}

స్థిర శూన్య repack_q5_K_tiled(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    స్థిరరాశి int64_t sb_చొప్పున_అడ్డువరుస = t->ne[0] / QK_K;
    స్థిరరాశి తార్కిక    use_neon   = (t->ne[1] % 8 == 0);
    repack_kquant_గమ్యం_tiled_grid<ఖండం_q5_K>(t, దత్తాంశం, స్థానభ్రంశం, size, HTP_TENSOR_TILE_కొలత_Q5_K,
        [sb_చొప్పున_అడ్డువరుస, use_neon](uint8_t * tile_dst, స్థిరరాశి ఖండం_q5_K * b, పూర్ణాంకం అడ్డువరుస, పూర్ణాంకం kt_local) {
            స్థిరరాశి దశాంశం d     = GGML_FP16_గమ్యం_FP32(b->d);
            స్థిరరాశి దశాంశం dmin  = GGML_FP16_గమ్యం_FP32(b->dmin);
            uint8_t * plane   = tile_dst + 640;

            uint8_t sc, m;
            get_scale_min_k4(kt_local, b->scales, &sc, &m);

            repack_k4_quants(tile_dst, b, అడ్డువరుస, kt_local, sb_చొప్పున_అడ్డువరుస, use_neon);

            స్థిరరాశి uint8_t hbit = (uint8_t) (1 << kt_local);
#ఒకవేళ defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q5_k_qh(b->qh, hbit, plane, అడ్డువరుస);
#లేకపోతే
            కొరకు (పూర్ణాంకం cp = 0; cp < 16; cp++) {
                స్థిరరాశి పూర్ణాంకం i    = cp / 4;
                స్థిరరాశి పూర్ణాంకం lane = (cp % 4) * 32 + అడ్డువరుస;
                ఒకవేళ (b->qh[2 * cp + 0] & hbit) {
                    plane[lane] |= (uint8_t) (1 << (2 * i));
                }
                ఒకవేళ (b->qh[2 * cp + 1] & hbit) {
                    plane[lane] |= (uint8_t) (1 << (2 * i + 1));
                }
            }
#అయితే_ముగింపు

            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
            scale_dst[2 * అడ్డువరుస + 0] = GGML_FP32_గమ్యం_FP16(d * (దశాంశం) sc);
            scale_dst[2 * అడ్డువరుస + 1] = GGML_FP32_గమ్యం_FP16(-dmin * (దశాంశం) m);
        });
}

స్థిర శూన్య repack_tiled_q5_K(శూన్య * దత్తాంశం, స్థిరరాశి ggml_tensor * t, size_t స్థానభ్రంశం, size_t size) {
    unpack_kquant_from_tiled_grid<ఖండం_q5_K>(దత్తాంశం, t, స్థానభ్రంశం, size, HTP_TENSOR_TILE_కొలత_Q5_K,
        [](ఖండం_q5_K * b, స్థిరరాశి uint8_t * sb_tiles, size_t tile_కొలత, పూర్ణాంకం అడ్డువరుస) {
            దశాంశం sub_scales[8];
            దశాంశం sub_mins[8];

            కొరకు (పూర్ణాంకం kt_local = 0; kt_local < 8; kt_local++) {
                స్థిరరాశి uint8_t *   tile_src  = sb_tiles + kt_local * tile_కొలత;
                స్థిరరాశి uint8_t *   plane     = tile_src + 640;
                స్థిరరాశి ggml_half * scale_src = (స్థిరరాశి ggml_half *) (tile_src + 512);

                uint8_t * qs_sub   = b->qs + (kt_local / 2) * 32;
                స్థిరరాశి పూర్ణాంకం జరుపు    = (kt_local & 1) ? 4 : 0;
                స్థిరరాశి uint8_t hbit = (uint8_t) (1 << kt_local);

                కొరకు (పూర్ణాంకం cp = 0; cp < 16; cp++) {
                    స్థిరరాశి uint8_t val = tile_src[cp * 32 + అడ్డువరుస];
                    qs_sub[2 * cp + 0] |= (uint8_t) ((val & 0x0F) << జరుపు);
                    qs_sub[2 * cp + 1] |= (uint8_t) ((val >> 4)   << జరుపు);

                    స్థిరరాశి పూర్ణాంకం i    = cp / 4;
                    స్థిరరాశి పూర్ణాంకం lane = (cp % 4) * 32 + అడ్డువరుస;
                    ఒకవేళ (plane[lane] & (1 << (2 * i))) {
                        b->qh[2 * cp + 0] |= hbit;
                    }
                    ఒకవేళ (plane[lane] & (1 << (2 * i + 1))) {
                        b->qh[2 * cp + 1] |= hbit;
                    }
                }

                స్థిరరాశి దశాంశం D = GGML_FP16_గమ్యం_FP32(scale_src[2 * అడ్డువరుస + 0]);
                స్థిరరాశి దశాంశం M = GGML_FP16_గమ్యం_FP32(scale_src[2 * అడ్డువరుస + 1]);
                sub_scales[kt_local] = (D > 0.0f) ? D : 0.0f;
                sub_mins[kt_local]   = (-M > 0.0f) ? -M : 0.0f;
            }

            pack_k4_scales_mins(b->scales, &b->d, &b->dmin, sub_scales, sub_mins);
        });
}

స్థిర అంతఃవరుస శూన్య q1_0_tiled_బిట్టు_location(పూర్ణాంకం k, పూర్ణాంకం అడ్డువరుస, పూర్ణాంకం & byte_స్థానభ్రంశం, పూర్ణాంకం & బిట్టు) {
    స్థిరరాశి పూర్ణాంకం quarter = k / 32;
    స్థిరరాశి పూర్ణాంకం kq      = k % 32;

    byte_స్థానభ్రంశం = quarter * 128 + 4 * అడ్డువరుస + (kq % 4);
    బిట్టు         = kq / 4;
}

స్థిర శూన్య repack_q1_0_tiled(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    స్థిరరాశి ఖండం_q1_0 * src_మాతృక = (స్థిరరాశి ఖండం_q1_0 *) దత్తాంశం;
    స్థిరరాశి int64_t ne0 = t->ne[0];
    స్థిరరాశి int64_t ne1 = t->ne[1];
    స్థిరరాశి int64_t ne2 = t->ne[2];
    స్థిరరాశి int64_t ne3 = t->ne[3];
    స్థిరరాశి పూర్ణాంకం n_నిలువు_tiles = hex_విడత_పైకి(ne1, 32) / 32;
    స్థిరరాశి పూర్ణాంకం n_k_tiles = ne0 / QK1_0;
    స్థిరరాశి size_t tile_కొలత = HTP_TENSOR_TILE_కొలత_Q1_0;
    స్థిరరాశి size_t మాతృక_కొలత = (size_t) n_నిలువు_tiles * n_k_tiles * tile_కొలత;
    స్థిరరాశి size_t ముక్క_కొలత = ne1 * ggml_అడ్డువరుస_కొలత(t->type, ne0);
    స్థిరరాశి int64_t ప్రారంభం_ముక్క = స్థానభ్రంశం / ముక్క_కొలత;
    స్థిరరాశి int64_t చివర_ముక్క = (std::min)((int64_t) ((స్థానభ్రంశం + size + ముక్క_కొలత - 1) / ముక్క_కొలత), ne2 * ne3);

    GGML_ASSERT(ne0 % QK1_0 == 0);

    కొరకు (int64_t ముక్క_idx = ప్రారంభం_ముక్క; ముక్క_idx < చివర_ముక్క; ముక్క_idx++) {
        స్థిరరాశి ఖండం_q1_0 * src_ముక్క = src_మాతృక + (ముక్క_idx - ప్రారంభం_ముక్క) * ne1 * n_k_tiles;
        uint8_t * మాతృక_dst = (uint8_t *) t->దత్తాంశం + ముక్క_idx * మాతృక_కొలత;

        repack_parallel_for_ct(మాతృక_కొలత, n_నిలువు_tiles, [&](పూర్ణాంకం ct0, పూర్ణాంకం ct1) {
            కొరకు (పూర్ణాంకం ct = ct0; ct < ct1; ct++) {
                కొరకు (పూర్ణాంకం kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = మాతృక_dst + ((size_t) ct * n_k_tiles + kt) * tile_కొలత;
                    ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
                    memset(tile_dst, 0, tile_కొలత);

                    కొరకు (పూర్ణాంకం అడ్డువరుస = 0; అడ్డువరుస < 32; అడ్డువరుస++) {
                        స్థిరరాశి int64_t r = ct * 32 + అడ్డువరుస;
                        ఒకవేళ (r >= ne1) {
                            కొనసాగు;
                        }

                        స్థిరరాశి ఖండం_q1_0 & ఖండం = src_ముక్క[r * n_k_tiles + kt];
                        scale_dst[అడ్డువరుస] = ఖండం.d;
#ఒకవేళ defined(__ARM_NEON) || defined(__aarch64__)
                        repack_neon_q1_0_ఖండం(ఖండం.qs, tile_dst, అడ్డువరుస);
#లేకపోతే
                        కొరకు (పూర్ణాంకం k = 0; k < QK1_0; k++) {
                            పూర్ణాంకం byte_స్థానభ్రంశం;
                            పూర్ణాంకం బిట్టు;
                            q1_0_tiled_బిట్టు_location(k, అడ్డువరుస, byte_స్థానభ్రంశం, బిట్టు);
                            ఒకవేళ (ఖండం.qs[k / 8] & (1u << (k % 8))) {
                                tile_dst[byte_స్థానభ్రంశం] |= (uint8_t) (1u << బిట్టు);
                            }
                        }
#అయితే_ముగింపు
                    }
                }
            }
        });
    }
}

స్థిర శూన్య repack_tiled_q1_0(శూన్య * దత్తాంశం, స్థిరరాశి ggml_tensor * t, size_t స్థానభ్రంశం, size_t size) {
    ఖండం_q1_0 * dst_మాతృక = (ఖండం_q1_0 *) దత్తాంశం;
    స్థిరరాశి int64_t ne0 = t->ne[0];
    స్థిరరాశి int64_t ne1 = t->ne[1];
    స్థిరరాశి int64_t ne2 = t->ne[2];
    స్థిరరాశి int64_t ne3 = t->ne[3];
    స్థిరరాశి పూర్ణాంకం n_నిలువు_tiles = hex_విడత_పైకి(ne1, 32) / 32;
    స్థిరరాశి పూర్ణాంకం n_k_tiles = ne0 / QK1_0;
    స్థిరరాశి size_t tile_కొలత = HTP_TENSOR_TILE_కొలత_Q1_0;
    స్థిరరాశి size_t మాతృక_కొలత = (size_t) n_నిలువు_tiles * n_k_tiles * tile_కొలత;
    స్థిరరాశి size_t అడ్డువరుస_కొలత_బైట్లు = ggml_అడ్డువరుస_కొలత(t->type, ne0);
    స్థిరరాశి size_t ముక్క_కొలత = ne1 * అడ్డువరుస_కొలత_బైట్లు;
    స్థిరరాశి int64_t ప్రారంభం_ముక్క = స్థానభ్రంశం / ముక్క_కొలత;
    స్థిరరాశి int64_t చివర_ముక్క = (std::min)((int64_t) ((స్థానభ్రంశం + size + ముక్క_కొలత - 1) / ముక్క_కొలత), ne2 * ne3);

    GGML_ASSERT(ne0 % QK1_0 == 0);

    కొరకు (int64_t ముక్క_idx = ప్రారంభం_ముక్క; ముక్క_idx < చివర_ముక్క; ముక్క_idx++) {
        స్థిరరాశి size_t cur_ప్రారంభం_byte = (std::max)(స్థానభ్రంశం, (size_t) ముక్క_idx * ముక్క_కొలత);
        స్థిరరాశి size_t cur_చివర_byte = (std::min)(స్థానభ్రంశం + size, (size_t) (ముక్క_idx + 1) * ముక్క_కొలత);
        స్థిరరాశి int64_t ప్రారంభం_అడ్డువరుస = (cur_ప్రారంభం_byte - (size_t) ముక్క_idx * ముక్క_కొలత) / అడ్డువరుస_కొలత_బైట్లు;
        స్థిరరాశి int64_t చివర_అడ్డువరుస = (std::min)((int64_t) ((cur_చివర_byte - (size_t) ముక్క_idx * ముక్క_కొలత + అడ్డువరుస_కొలత_బైట్లు - 1) / అడ్డువరుస_కొలత_బైట్లు), ne1);
        ఖండం_q1_0 * dst_ముక్క = dst_మాతృక + (cur_ప్రారంభం_byte - స్థానభ్రంశం) / పరిమాణం(ఖండం_q1_0);
        స్థిరరాశి uint8_t * మాతృక_src = (స్థిరరాశి uint8_t *) t->దత్తాంశం + ముక్క_idx * మాతృక_కొలత;

        కొరకు (పూర్ణాంకం ct = ప్రారంభం_అడ్డువరుస / 32; ct < (చివర_అడ్డువరుస + 31) / 32; ct++) {
            కొరకు (పూర్ణాంకం kt = 0; kt < n_k_tiles; kt++) {
                స్థిరరాశి uint8_t * tile_src = మాతృక_src + ((size_t) ct * n_k_tiles + kt) * tile_కొలత;
                స్థిరరాశి ggml_half * scale_src = (స్థిరరాశి ggml_half *) (tile_src + 512);

                కొరకు (పూర్ణాంకం అడ్డువరుస = 0; అడ్డువరుస < 32; అడ్డువరుస++) {
                    స్థిరరాశి int64_t r = ct * 32 + అడ్డువరుస;
                    ఒకవేళ (r < ప్రారంభం_అడ్డువరుస || r >= చివర_అడ్డువరుస) {
                        కొనసాగు;
                    }

                    ఖండం_q1_0 & ఖండం = dst_ముక్క[(r - ప్రారంభం_అడ్డువరుస) * n_k_tiles + kt];
                    memset(ఖండం.qs, 0, పరిమాణం(ఖండం.qs));
                    ఖండం.d = scale_src[అడ్డువరుస];
                    కొరకు (పూర్ణాంకం k = 0; k < QK1_0; k++) {
                        పూర్ణాంకం byte_స్థానభ్రంశం;
                        పూర్ణాంకం బిట్టు;
                        q1_0_tiled_బిట్టు_location(k, అడ్డువరుస, byte_స్థానభ్రంశం, బిట్టు);
                        ఒకవేళ (tile_src[byte_స్థానభ్రంశం] & (1u << బిట్టు)) {
                            ఖండం.qs[k / 8] |= (uint8_t) (1u << (k % 8));
                        }
                    }
                }
            }
        }
    }
}

// --- Top-level Repack / Unpack Dispatchers ---

శూన్య htp_repack_tensor_tiled(ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t size) {
    ఎంపిక (tensor->type) {
        స్థితి GGML_TYPE_Q1_0:
            repack_q1_0_tiled(tensor, దత్తాంశం, 0, size);
            ఆపు;
        స్థితి GGML_TYPE_Q4_0:
        స్థితి GGML_TYPE_IQ4_NL:
            repack_q4_0_tiled(tensor, దత్తాంశం, 0, size);
            ఆపు;
        స్థితి GGML_TYPE_Q4_1:
            repack_q4_1_tiled(tensor, దత్తాంశం, 0, size);
            ఆపు;
        స్థితి GGML_TYPE_Q8_0:
            repack_q8_0_tiled(tensor, దత్తాంశం, 0, size);
            ఆపు;
        స్థితి GGML_TYPE_MXFP4:
            repack_mxfp4_tiled(tensor, దత్తాంశం, 0, size);
            ఆపు;
        స్థితి GGML_TYPE_Q2_0:
            repack_q2_0_tiled(tensor, దత్తాంశం, 0, size);
            ఆపు;
        స్థితి GGML_TYPE_Q2_K:
            repack_q2_K_tiled(tensor, దత్తాంశం, 0, size);
            ఆపు;
        స్థితి GGML_TYPE_Q3_K:
            repack_q3_K_tiled(tensor, దత్తాంశం, 0, size);
            ఆపు;
        స్థితి GGML_TYPE_Q4_K:
            repack_q4_K_tiled(tensor, దత్తాంశం, 0, size);
            ఆపు;
        స్థితి GGML_TYPE_Q5_K:
            repack_q5_K_tiled(tensor, దత్తాంశం, 0, size);
            ఆపు;
        స్థితి GGML_TYPE_Q6_K:
            repack_q6_K_tiled(tensor, దత్తాంశం, 0, size);
            ఆపు;
        స్థితి GGML_TYPE_F16:
            repack_float_గమ్యం_f16_tiled<uint16_t>(tensor, దత్తాంశం, 0, size,
                [](uint16_t v) { తిరిగి v; });
            ఆపు;
        స్థితి GGML_TYPE_F32:
            repack_float_గమ్యం_f16_tiled<దశాంశం>(tensor, దత్తాంశం, 0, size,
                [](దశాంశం v) { తిరిగి (uint16_t) GGML_FP32_గమ్యం_FP16(v); });
            ఆపు;
        స్థితి GGML_TYPE_BF16:
            repack_float_గమ్యం_f16_tiled<ggml_bf16_t>(tensor, దత్తాంశం, 0, size,
                [](ggml_bf16_t v) { తిరిగి (uint16_t) GGML_FP32_గమ్యం_FP16(GGML_BF16_గమ్యం_FP32(v)); });
            ఆపు;
        అప్రమేయ:
            ఆపు;
    }
}

శూన్య htp_unpack_tensor_tiled(శూన్య * దత్తాంశం, స్థిరరాశి ggml_tensor * tensor, size_t స్థానభ్రంశం, size_t size) {
    GGML_ASSERT(స్థానభ్రంశం + size <= ggml_nbytes(tensor));

    ఎంపిక (tensor->type) {
        స్థితి GGML_TYPE_Q1_0:
            repack_tiled_q1_0(దత్తాంశం, tensor, స్థానభ్రంశం, size);
            ఆపు;
        స్థితి GGML_TYPE_Q4_0:
        స్థితి GGML_TYPE_IQ4_NL:
            repack_tiled_q4_0(దత్తాంశం, tensor, స్థానభ్రంశం, size);
            ఆపు;
        స్థితి GGML_TYPE_Q4_1:
            repack_tiled_q4_1(దత్తాంశం, tensor, స్థానభ్రంశం, size);
            ఆపు;
        స్థితి GGML_TYPE_Q8_0:
            repack_tiled_q8_0(దత్తాంశం, tensor, స్థానభ్రంశం, size);
            ఆపు;
        స్థితి GGML_TYPE_MXFP4:
            repack_tiled_mxfp4(దత్తాంశం, tensor, స్థానభ్రంశం, size);
            ఆపు;
        స్థితి GGML_TYPE_Q2_0:
            repack_tiled_q2_0(దత్తాంశం, tensor, స్థానభ్రంశం, size);
            ఆపు;
        స్థితి GGML_TYPE_Q2_K:
            repack_tiled_q2_K(దత్తాంశం, tensor, స్థానభ్రంశం, size);
            ఆపు;
        స్థితి GGML_TYPE_Q3_K:
            repack_tiled_q3_K(దత్తాంశం, tensor, స్థానభ్రంశం, size);
            ఆపు;
        స్థితి GGML_TYPE_Q4_K:
            repack_tiled_q4_K(దత్తాంశం, tensor, స్థానభ్రంశం, size);
            ఆపు;
        స్థితి GGML_TYPE_Q5_K:
            repack_tiled_q5_K(దత్తాంశం, tensor, స్థానభ్రంశం, size);
            ఆపు;
        స్థితి GGML_TYPE_Q6_K:
            repack_tiled_q6_K(దత్తాంశం, tensor, స్థానభ్రంశం, size);
            ఆపు;
        స్థితి GGML_TYPE_F16:
            unpack_f16_tiled_గమ్యం_float<uint16_t>(దత్తాంశం, tensor, స్థానభ్రంశం, size,
                [](uint16_t v) { తిరిగి v; });
            ఆపు;
        స్థితి GGML_TYPE_F32:
            unpack_f16_tiled_గమ్యం_float<దశాంశం>(దత్తాంశం, tensor, స్థానభ్రంశం, size,
                [](uint16_t v) { తిరిగి ggml_fp16_గమ్యం_fp32((ggml_fp16_t) v); });
            ఆపు;
        స్థితి GGML_TYPE_BF16:
            unpack_f16_tiled_గమ్యం_float<ggml_bf16_t>(దత్తాంశం, tensor, స్థానభ్రంశం, size,
                [](uint16_t v) { తిరిగి ggml_fp32_గమ్యం_bf16(ggml_fp16_గమ్యం_fp32((ggml_fp16_t) v)); });
            ఆపు;
        అప్రమేయ:
            జ్ఞాపకం_నకలు(దత్తాంశం, (స్థిరరాశి అక్షరం *) tensor->దత్తాంశం + స్థానభ్రంశం, size);
            ఆపు;
    }
}
