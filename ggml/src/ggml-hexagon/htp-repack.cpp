#অন্তর্ভাব "htp-repack.h"

#define GGML_COMMON_IMPL_CPP
#অন্তর্ভাব "ggml-common.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "ggml-quants.h"

#অন্তর্ভাব <algorithm>
#অন্তর্ভাব <cmath>
#অন্তর্ভাব <cstring>
#অন্তর্ভাব <initializer_list>
#অন্তর্ভাব <thread>

#যদি defined(__ARM_NEON) || defined(__aarch64__)
#    include <arm_neon.h>
#endif

// --- 4-bit block-32 nibble helpers (shared by Q4_0, IQ4_NL, Q4_1, MXFP4) ---

স্থির অন্তঃরেখ শূন্য unpack_nibbles_32(uint8_t * dst, ধ্রুবক uint8_t * qs) {
    জন্য (unsigned পূর্ণসংখ্যা i = 0; i < 16; ++i) {
        dst[i + 0]  = qs[i] & 0x0F;
        dst[i + 16] = qs[i] >> 4;
    }
}

স্থির অন্তঃরেখ শূন্য pack_nibbles_32(uint8_t * qs, ধ্রুবক uint8_t * src) {
    জন্য (unsigned পূর্ণসংখ্যা i = 0; i < 16; ++i) {
        qs[i] = src[i + 0] | (src[i + 16] << 4);
    }
}

স্থির অন্তঃরেখ শূন্য transpose_nibbles_to_tile(uint8_t * tile_dst, ধ্রুবক uint8_t tile_quants[32][32]) {
    জন্য (পূর্ণসংখ্যা cp = 0; cp < 16; cp++) {
        জন্য (পূর্ণসংখ্যা row = 0; row < 32; row++) {
            tile_dst[cp * 32 + row] = (tile_quants[row][2 * cp + 1] << 4) | tile_quants[row][2 * cp];
        }
    }
}

স্থির অন্তঃরেখ শূন্য transpose_tile_to_nibbles(uint8_t tile_quants[32][32], ধ্রুবক uint8_t * tile_src) {
    জন্য (পূর্ণসংখ্যা cp = 0; cp < 16; cp++) {
        জন্য (পূর্ণসংখ্যা row = 0; row < 32; row++) {
            ধ্রুবক uint8_t val = tile_src[cp * 32 + row];
            tile_quants[row][2 * cp + 0] = val & 0x0F;
            tile_quants[row][2 * cp + 1] = val >> 4;
        }
    }
}

#যদি defined(__ARM_NEON) || defined(__aarch64__)
স্থির শূন্য repack_neon_transpose8x8(ধ্রুবক uint8x8_t rows[8], uint8x8_t columns[8]) {
    ধ্রুবক uint8x8x2_t t0 = vtrn_u8(rows[0], rows[1]);
    ধ্রুবক uint8x8x2_t t1 = vtrn_u8(rows[2], rows[3]);
    ধ্রুবক uint8x8x2_t t2 = vtrn_u8(rows[4], rows[5]);
    ধ্রুবক uint8x8x2_t t3 = vtrn_u8(rows[6], rows[7]);
    ধ্রুবক uint16x4x2_t t4 = vtrn_u16(vreinterpret_u16_u8(t0.val[0]), vreinterpret_u16_u8(t1.val[0]));
    ধ্রুবক uint16x4x2_t t5 = vtrn_u16(vreinterpret_u16_u8(t0.val[1]), vreinterpret_u16_u8(t1.val[1]));
    ধ্রুবক uint16x4x2_t t6 = vtrn_u16(vreinterpret_u16_u8(t2.val[0]), vreinterpret_u16_u8(t3.val[0]));
    ধ্রুবক uint16x4x2_t t7 = vtrn_u16(vreinterpret_u16_u8(t2.val[1]), vreinterpret_u16_u8(t3.val[1]));
    ধ্রুবক uint32x2x2_t t8 = vtrn_u32(vreinterpret_u32_u16(t4.val[0]), vreinterpret_u32_u16(t6.val[0]));
    ধ্রুবক uint32x2x2_t t9 = vtrn_u32(vreinterpret_u32_u16(t4.val[1]), vreinterpret_u32_u16(t6.val[1]));
    ধ্রুবক uint32x2x2_t ta = vtrn_u32(vreinterpret_u32_u16(t5.val[0]), vreinterpret_u32_u16(t7.val[0]));
    ধ্রুবক uint32x2x2_t tb = vtrn_u32(vreinterpret_u32_u16(t5.val[1]), vreinterpret_u32_u16(t7.val[1]));
    columns[0] = vreinterpret_u8_u32(t8.val[0]);
    columns[1] = vreinterpret_u8_u32(ta.val[0]);
    columns[2] = vreinterpret_u8_u32(t9.val[0]);
    columns[3] = vreinterpret_u8_u32(tb.val[0]);
    columns[4] = vreinterpret_u8_u32(t8.val[1]);
    columns[5] = vreinterpret_u8_u32(ta.val[1]);
    columns[6] = vreinterpret_u8_u32(t9.val[1]);
    columns[7] = vreinterpret_u8_u32(tb.val[1]);
}

স্থির uint8x8_t repack_neon_pack_4bit_pairs(uint8x8_t first, uint8x8_t second, যৌক্তিক উপরের) {
    ধ্রুবক uint8x8x2_t separated = vuzp_u8(first, second);
    ধ্রুবক uint8x8_t mask = vdup_n_u8(0x0F);
    uint8x8_t even = separated.val[0];
    uint8x8_t odd  = separated.val[1];

    যদি (উপরের) {
        even = vshr_n_u8(even, 4);
        odd  = vshr_n_u8(odd, 4);
    }

    even = vand_u8(even, mask);
    odd  = vand_u8(odd, mask);
    ফেরত vorr_u8(vshl_n_u8(odd, 4), even);
}

স্থির শূন্য repack_neon_4bit_rows(ধ্রুবক uint8_t * qs[8], uint8_t * tile_dst, পূর্ণসংখ্যা row0) {
    uint8x8_t নিচের_rows[8];
    uint8x8_t উপরের_rows[8];
    uint8x8_t columns[8];

    জন্য (পূর্ণসংখ্যা rr = 0; rr < 8; ++rr) {
        ধ্রুবক uint8x16_t packed = vld1q_u8(qs[rr]);
        ধ্রুবক uint8x8_t first = vget_নিচের_u8(packed);
        ধ্রুবক uint8x8_t second = vget_উপরের_u8(packed);
        নিচের_rows[rr]  = repack_neon_pack_4bit_pairs(first, second, মিথ্যা);
        উপরের_rows[rr] = repack_neon_pack_4bit_pairs(first, second, সত্য);
    }

    repack_neon_transpose8x8(নিচের_rows, columns);
    জন্য (পূর্ণসংখ্যা cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + cp * 32 + row0, columns[cp]);
    }

    repack_neon_transpose8x8(উপরের_rows, columns);
    জন্য (পূর্ণসংখ্যা cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + (cp + 8) * 32 + row0, columns[cp]);
    }
}

স্থির শূন্য repack_neon_k4_rows(ধ্রুবক uint8_t * qs[8], uint8_t * tile_dst, পূর্ণসংখ্যা row0, যৌক্তিক উপরের) {
    uint8x8_t first_rows[8];
    uint8x8_t second_rows[8];
    uint8x8_t columns[8];

    জন্য (পূর্ণসংখ্যা rr = 0; rr < 8; ++rr) {
        first_rows[rr] = repack_neon_pack_4bit_pairs(vld1_u8(qs[rr]), vld1_u8(qs[rr] + 8), উপরের);
        second_rows[rr] = repack_neon_pack_4bit_pairs(vld1_u8(qs[rr] + 16), vld1_u8(qs[rr] + 24), উপরের);
    }

    repack_neon_transpose8x8(first_rows, columns);
    জন্য (পূর্ণসংখ্যা cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + cp * 32 + row0, columns[cp]);
    }

    repack_neon_transpose8x8(second_rows, columns);
    জন্য (পূর্ণসংখ্যা cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + (cp + 8) * 32 + row0, columns[cp]);
    }
}

স্থির শূন্য repack_neon_q8_rows(ধ্রুবক uint8_t * qs[8], uint8_t * tile_dst, পূর্ণসংখ্যা row0) {
    uint8x8_t rows[8];
    uint8x8_t columns[8];

    জন্য (পূর্ণসংখ্যা cp = 0; cp < 16; cp += 4) {
        জন্য (পূর্ণসংখ্যা rr = 0; rr < 8; ++rr) {
            rows[rr] = vld1_u8(qs[rr] + 2 * cp);
        }
        repack_neon_transpose8x8(rows, columns);

        জন্য (পূর্ণসংখ্যা pair = 0; pair < 4; ++pair) {
            ধ্রুবক uint8x8x2_t interleaved = vzip_u8(columns[2 * pair], columns[2 * pair + 1]);
            uint8_t * dst = tile_dst + (cp + pair) * 64 + 2 * row0;
            vst1_u8(dst, interleaved.val[0]);
            vst1_u8(dst + 8, interleaved.val[1]);
        }
    }
}

স্থির অন্তঃরেখ শূন্য repack_store_u32(uint8_t * dst, uint32_t v) {
    memcpy(dst, &v, আকার(v));
}

স্থির অন্তঃরেখ uint32x4_t repack_neon_transpose_2bit_4x4(uint32x4_t w) {
    uint32x4_t t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 6)), vdupq_n_u32(0x00CC00CCu));
    w = veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 6)));
    t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 12)), vdupq_n_u32(0x0000F0F0u));
    ফেরত veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 12)));
}

স্থির অন্তঃরেখ শূন্য repack_neon_q2_0_block(ধ্রুবক uint8_t * qs, uint8_t * tile, পূর্ণসংখ্যা row) {
    ধ্রুবক uint32x4_t w = repack_neon_transpose_2bit_4x4(vreinterpretq_u32_u8(vld1q_u8(qs)));
    repack_store_u32(tile + 0 * 128 + 4 * row, vgetq_lane_u32(w, 0));
    repack_store_u32(tile + 1 * 128 + 4 * row, vgetq_lane_u32(w, 1));
    repack_store_u32(tile + 2 * 128 + 4 * row, vgetq_lane_u32(w, 2));
    repack_store_u32(tile + 3 * 128 + 4 * row, vgetq_lane_u32(w, 3));
}

স্থির অন্তঃরেখ শূন্য repack_neon_q1_0_block(ধ্রুবক uint8_t * qs, uint8_t * tile, পূর্ণসংখ্যা row) {
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

স্থির অন্তঃরেখ শূন্য repack_neon_pack_2bit_plane(uint8x16_t v0, uint8x16_t v1, uint8_t * dst, পূর্ণসংখ্যা row) {
    ধ্রুবক uint32x4x2_t z   = vuzpq_u32(vreinterpretq_u32_u8(v0), vreinterpretq_u32_u8(v1));
    ধ্রুবক uint32x4_t   p01 = vorrq_u32(z.val[0], vshlq_n_u32(z.val[1], 2));
    ধ্রুবক uint32x2x2_t u   = vuzp_u32(vget_নিচের_u32(p01), vget_উপরের_u32(p01));
    ধ্রুবক uint32x2_t   out = vorr_u32(u.val[0], vshl_n_u32(u.val[1], 4));
    repack_store_u32(dst + 0 * 128 + row * 4, vget_lane_u32(out, 0));
    repack_store_u32(dst + 1 * 128 + row * 4, vget_lane_u32(out, 1));
}

স্থির অন্তঃরেখ শূন্য repack_neon_q2_3_k_low2(ধ্রুবক uint8_t * qs, পূর্ণসংখ্যা kt_local, uint8_t * lo_pl, পূর্ণসংখ্যা row) {
    ধ্রুবক uint8_t *  qs_sub = qs + (kt_local / 4) * 32;
    ধ্রুবক int8x16_t  shift  = vdupq_n_s8(-2 * (kt_local % 4));
    ধ্রুবক uint8x16_t mask   = vdupq_n_u8(0x03);
    ধ্রুবক uint8x16_t v0     = vandq_u8(vshlq_u8(vld1q_u8(qs_sub +  0), shift), mask);
    ধ্রুবক uint8x16_t v1     = vandq_u8(vshlq_u8(vld1q_u8(qs_sub + 16), shift), mask);
    repack_neon_pack_2bit_plane(v0, v1, lo_pl, row);
}

স্থির অন্তঃরেখ শূন্য repack_neon_q3_k_hmask(ধ্রুবক uint8_t * hmask, পূর্ণসংখ্যা kt_local, uint8_t * neg_pl, পূর্ণসংখ্যা row) {
    স্থির ধ্রুবক uint32_t mask_শব্দগুলি[4] = { 0x01010101u, 0x02020202u, 0x04040404u, 0x08080808u };
    ধ্রুবক uint8x16_t m0   = vreinterpretq_u8_u32(vld1q_u32(mask_শব্দগুলি));
    ধ্রুবক uint8x16_t m1   = vshlq_n_u8(m0, 4);
    ধ্রুবক uint8x16_t hbit = vdupq_n_u8((uint8_t) (1u << kt_local));
    ধ্রুবক uint8x16_t b0   = vbicq_u8(m0, vtstq_u8(vld1q_u8(hmask +  0), hbit));
    ধ্রুবক uint8x16_t b1   = vbicq_u8(m1, vtstq_u8(vld1q_u8(hmask + 16), hbit));
    ধ্রুবক uint32x4_t b01  = vreinterpretq_u32_u8(vorrq_u8(b0, b1));
    ধ্রুবক uint32x2_t f0   = vorr_u32(vget_নিচের_u32(b01), vget_উপরের_u32(b01));
    ধ্রুবক uint32x2_t f1   = vpadd_u32(f0, f0);
    repack_store_u32(neg_pl + row * 4, vget_lane_u32(f1, 0));
}

স্থির অন্তঃরেখ শূন্য repack_neon_q6_k(ধ্রুবক block_q6_K * b, পূর্ণসংখ্যা kt_local, uint8_t * lo_pl, uint8_t * hi_pl, পূর্ণসংখ্যা row) {
    ধ্রুবক পূর্ণসংখ্যা        c      = kt_local / 4;
    ধ্রুবক পূর্ণসংখ্যা        g_sb   = kt_local % 4;
    ধ্রুবক uint8_t *  ql     = b->ql + c * 64 + (g_sb & 1) * 32;
    ধ্রুবক uint8_t *  qh     = b->qh + c * 32;
    ধ্রুবক int8x16_t  lshift = vdupq_n_s8((g_sb >= 2) ? -4 : 0);
    ধ্রুবক int8x16_t  hshift = vdupq_n_s8(-2 * g_sb);
    ধ্রুবক uint8x16_t mask4  = vdupq_n_u8(0x0F);
    ধ্রুবক uint8x16_t mask2  = vdupq_n_u8(0x03);

    ধ্রুবক uint8x16_t lo0  = vandq_u8(vshlq_u8(vld1q_u8(ql +  0), lshift), mask4);
    ধ্রুবক uint8x16_t lo1  = vandq_u8(vshlq_u8(vld1q_u8(ql + 16), lshift), mask4);
    ধ্রুবক uint32x4x2_t lz = vuzpq_u32(vreinterpretq_u32_u8(lo0), vreinterpretq_u32_u8(lo1));
    ধ্রুবক uint32x4_t lout = vorrq_u32(lz.val[0], vshlq_n_u32(lz.val[1], 4));
    repack_store_u32(lo_pl + 0 * 128 + row * 4, vgetq_lane_u32(lout, 0));
    repack_store_u32(lo_pl + 1 * 128 + row * 4, vgetq_lane_u32(lout, 1));
    repack_store_u32(lo_pl + 2 * 128 + row * 4, vgetq_lane_u32(lout, 2));
    repack_store_u32(lo_pl + 3 * 128 + row * 4, vgetq_lane_u32(lout, 3));

    ধ্রুবক uint8x16_t hi0 = vandq_u8(vshlq_u8(vld1q_u8(qh +  0), hshift), mask2);
    ধ্রুবক uint8x16_t hi1 = vandq_u8(vshlq_u8(vld1q_u8(qh + 16), hshift), mask2);
    repack_neon_pack_2bit_plane(hi0, hi1, hi_pl, row);
}

স্থির অন্তঃরেখ শূন্য repack_neon_q5_k_qh(ধ্রুবক uint8_t * qh, uint8_t hbit, uint8_t * plane, পূর্ণসংখ্যা row) {
    স্থির ধ্রুবক uint8_t mask_bytes[16] = { 1, 2, 1, 2, 1, 2, 1, 2, 4, 8, 4, 8, 4, 8, 4, 8 };
    ধ্রুবক uint8x16_t m0   = vld1q_u8(mask_bytes);
    ধ্রুবক uint8x16_t m1   = vshlq_n_u8(m0, 4);
    ধ্রুবক uint8x16_t vbit = vdupq_n_u8(hbit);
    ধ্রুবক uint8x16_t b0   = vandq_u8(vtstq_u8(vld1q_u8(qh +  0), vbit), m0);
    ধ্রুবক uint8x16_t b1   = vandq_u8(vtstq_u8(vld1q_u8(qh + 16), vbit), m1);
    ধ্রুবক uint8x16_t b01  = vorrq_u8(b0, b1);
    ধ্রুবক uint8x8_t  fold = vorr_u8(vget_নিচের_u8(b01), vget_উপরের_u8(b01));
    ধ্রুবক uint8x8_t  out  = vpadd_u8(fold, fold);
    plane[0 * 32 + row] = vget_lane_u8(out, 0);
    plane[1 * 32 + row] = vget_lane_u8(out, 1);
    plane[2 * 32 + row] = vget_lane_u8(out, 2);
    plane[3 * 32 + row] = vget_lane_u8(out, 3);
}

স্থির অন্তঃরেখ uint16x8_t repack_neon_load_f16x8(ধ্রুবক uint16_t * src) {
    ফেরত vld1q_u16(src);
}

#যদি defined(__aarch64__) || defined(__ARM_FP16_FORMAT_IEEE)
স্থির অন্তঃরেখ uint16x8_t repack_neon_load_f16x8(ধ্রুবক দশমিক * src) {
    ধ্রুবক float16x4_t lo = vcvt_f16_f32(vld1q_f32(src));
    ধ্রুবক float16x4_t hi = vcvt_f16_f32(vld1q_f32(src + 4));
    ফেরত vreinterpretq_u16_f16(vcombine_f16(lo, hi));
}

স্থির অন্তঃরেখ uint16x8_t repack_neon_load_f16x8(ধ্রুবক ggml_bf16_t * src) {
    ধ্রুবক uint16x8_t  bf   = vld1q_u16((ধ্রুবক uint16_t *) src);
    ধ্রুবক float32x4_t f_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_নিচের_u16(bf), 16));
    ধ্রুবক float32x4_t f_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_উপরের_u16(bf), 16));
    ধ্রুবক float16x4_t lo   = vcvt_f16_f32(f_lo);
    ধ্রুবক float16x4_t hi   = vcvt_f16_f32(f_hi);
    ফেরত vreinterpretq_u16_f16(vcombine_f16(lo, hi));
}
#অন্যথা
স্থির অন্তঃরেখ uint16x8_t repack_neon_load_f16x8(ধ্রুবক দশমিক * src) {
    uint16_t tmp[8];
    জন্য (পূর্ণসংখ্যা i = 0; i < 8; ++i) {
        tmp[i] = GGML_FP32_TO_FP16(src[i]);
    }
    ফেরত vld1q_u16(tmp);
}

স্থির অন্তঃরেখ uint16x8_t repack_neon_load_f16x8(ধ্রুবক ggml_bf16_t * src) {
    uint16_t tmp[8];
    জন্য (পূর্ণসংখ্যা i = 0; i < 8; ++i) {
        tmp[i] = GGML_FP32_TO_FP16(GGML_BF16_TO_FP32(src[i]));
    }
    ফেরত vld1q_u16(tmp);
}
#endif

template <typename SrcT>
স্থির শূন্য repack_neon_f16_tile(uint16_t * tile_dst, ধ্রুবক SrcT * src_slice, int64_t row_stride, পূর্ণসংখ্যা ct, পূর্ণসংখ্যা kt) {
    জন্য (পূর্ণসংখ্যা row0 = 0; row0 < 32; row0 += 4) {
        ধ্রুবক SrcT * r0 = src_slice + (int64_t) (ct * 32 + row0 + 0) * row_stride + kt * 32;
        ধ্রুবক SrcT * r1 = src_slice + (int64_t) (ct * 32 + row0 + 1) * row_stride + kt * 32;
        ধ্রুবক SrcT * r2 = src_slice + (int64_t) (ct * 32 + row0 + 2) * row_stride + kt * 32;
        ধ্রুবক SrcT * r3 = src_slice + (int64_t) (ct * 32 + row0 + 3) * row_stride + kt * 32;
        জন্য (পূর্ণসংখ্যা cp = 0; cp < 16; cp += 4) {
            ধ্রুবক uint32x4_t v0 = vreinterpretq_u32_u16(repack_neon_load_f16x8(r0 + 2 * cp));
            ধ্রুবক uint32x4_t v1 = vreinterpretq_u32_u16(repack_neon_load_f16x8(r1 + 2 * cp));
            ধ্রুবক uint32x4_t v2 = vreinterpretq_u32_u16(repack_neon_load_f16x8(r2 + 2 * cp));
            ধ্রুবক uint32x4_t v3 = vreinterpretq_u32_u16(repack_neon_load_f16x8(r3 + 2 * cp));

            ধ্রুবক uint32x2x2_t t0 = vtrn_u32(vget_নিচের_u32(v0),  vget_নিচের_u32(v1));
            ধ্রুবক uint32x2x2_t t1 = vtrn_u32(vget_উপরের_u32(v0), vget_উপরের_u32(v1));
            ধ্রুবক uint32x2x2_t t2 = vtrn_u32(vget_নিচের_u32(v2),  vget_নিচের_u32(v3));
            ধ্রুবক uint32x2x2_t t3 = vtrn_u32(vget_উপরের_u32(v2), vget_উপরের_u32(v3));

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
স্থির শূন্য repack_parallel_for_ct(size_t slice_bytes, পূর্ণসংখ্যা n_col_tiles, Fn && fn) {
    পূর্ণসংখ্যা n_threads = 1;
    যদি (slice_bytes > 128 * 1024) {
        জন্য (পূর্ণসংখ্যা nt : { 4, 3, 2 }) {
            যদি (n_col_tiles % nt == 0) {
                n_threads = nt;
                বিরতি;
            }
        }
    }

    যদি (n_threads <= 1) {
        fn(0, n_col_tiles);
        ফেরত;
    }

    ধ্রুবক পূর্ণসংখ্যা chunk = n_col_tiles / n_threads;
    std::thread workers[3];
    জন্য (পূর্ণসংখ্যা i = 0; i < n_threads - 1; ++i) {
        workers[i] = std::thread([&fn, i, chunk]() {
            fn(i * chunk, (i + 1) * chunk);
        });
    }
    fn((n_threads - 1) * chunk, n_col_tiles);
    জন্য (পূর্ণসংখ্যা i = 0; i < n_threads - 1; ++i) {
        workers[i].join();
    }
}

template <typename SrcElemT, পূর্ণসংখ্যা ELEMS_PER_KTILE, typename PackTileFn>
স্থির শূন্য repack_to_tiled_grid_32(ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t offset, size_t size,
                                    size_t tile_আকার, PackTileFn && pack_tile) {
    ধ্রুবক SrcElemT * src_matrix = (ধ্রুবক SrcElemT *) তথ্য;
    ধ্রুবক int64_t ne0 = t->ne[0];
    ধ্রুবক int64_t ne1 = t->ne[1];
    ধ্রুবক int64_t ne2 = t->ne[2];
    ধ্রুবক int64_t ne3 = t->ne[3];
    ধ্রুবক পূর্ণসংখ্যা n_col_tiles = hex_round_up(ne1, 32) / 32;
    ধ্রুবক পূর্ণসংখ্যা n_k_tiles   = hex_round_up(ne0, 32) / 32;
    ধ্রুবক size_t matrix_আকার = (size_t) n_col_tiles * n_k_tiles * tile_আকার;

    ধ্রুবক size_t slice_আকার = (size_t) ne1 * ggml_row_আকার(t->type, ne0);
    ধ্রুবক int64_t start_slice = offset / slice_আকার;
    int64_t end_slice = (offset + size + slice_আকার - 1) / slice_আকার;
    যদি (end_slice > ne2 * ne3) {
        end_slice = ne2 * ne3;
    }

    ধ্রুবক int64_t row_stride = (ne0 / (32 / ELEMS_PER_KTILE));

    জন্য (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        ধ্রুবক SrcElemT * src_slice = src_matrix + (slice_idx - start_slice) * (ne1 * row_stride);
        uint8_t * matrix_dst = (uint8_t *) t->তথ্য + slice_idx * matrix_আকার;

        repack_parallel_for_ct(matrix_আকার, n_col_tiles, [&](পূর্ণসংখ্যা ct0, পূর্ণসংখ্যা ct1) {
            জন্য (পূর্ণসংখ্যা ct = ct0; ct < ct1; ct++) {
                জন্য (পূর্ণসংখ্যা kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = matrix_dst + ((size_t) ct * n_k_tiles + kt) * tile_আকার;
                    pack_tile(tile_dst, src_slice, ne0, ne1, row_stride, ct, kt);
                }
            }
        });
    }
}

template <typename DstElemT, পূর্ণসংখ্যা ELEMS_PER_KTILE, typename UnpackTileFn>
স্থির শূন্য unpack_from_tiled_grid_32(শূন্য * তথ্য, ধ্রুবক ggml_tensor * t, size_t offset, size_t size,
                                      size_t tile_আকার, UnpackTileFn && unpack_tile) {
    uint8_t * dst_bytes = (uint8_t *) তথ্য;
    ধ্রুবক int64_t ne0 = t->ne[0];
    ধ্রুবক int64_t ne1 = t->ne[1];
    ধ্রুবক int64_t ne2 = t->ne[2];
    ধ্রুবক int64_t ne3 = t->ne[3];
    ধ্রুবক পূর্ণসংখ্যা n_col_tiles = hex_round_up(ne1, 32) / 32;
    ধ্রুবক পূর্ণসংখ্যা n_k_tiles   = hex_round_up(ne0, 32) / 32;
    ধ্রুবক size_t matrix_আকার = (size_t) n_col_tiles * n_k_tiles * tile_আকার;

    ধ্রুবক size_t row_আকার_bytes = ggml_row_আকার(t->type, ne0);
    ধ্রুবক size_t slice_আকার     = (size_t) ne1 * row_আকার_bytes;
    ধ্রুবক int64_t start_slice   = offset / slice_আকার;
    int64_t end_slice           = (offset + size + slice_আকার - 1) / slice_আকার;
    যদি (end_slice > ne2 * ne3) {
        end_slice = ne2 * ne3;
    }

    ধ্রুবক int64_t row_stride = (ne0 / (32 / ELEMS_PER_KTILE));

    জন্য (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        ধ্রুবক size_t cur_start_byte     = (std::max)(offset, (size_t) slice_idx * slice_আকার);
        ধ্রুবক size_t cur_end_byte       = (std::min)(offset + size, (size_t) (slice_idx + 1) * slice_আকার);
        ধ্রুবক size_t slice_offset_start = cur_start_byte - (size_t) slice_idx * slice_আকার;
        ধ্রুবক size_t slice_offset_end   = cur_end_byte - (size_t) slice_idx * slice_আকার;

        ধ্রুবক int64_t start_row = slice_offset_start / row_আকার_bytes;
        ধ্রুবক int64_t end_row   = (std::min)((int64_t) ((slice_offset_end + row_আকার_bytes - 1) / row_আকার_bytes), ne1);

        ধ্রুবক পূর্ণসংখ্যা start_ct = (পূর্ণসংখ্যা) (start_row / 32);
        ধ্রুবক পূর্ণসংখ্যা end_ct   = (std::min)((পূর্ণসংখ্যা) ((end_row + 31) / 32), n_col_tiles);

        DstElemT * dst_slice = (DstElemT *) (dst_bytes + (cur_start_byte - offset));
        ধ্রুবক uint8_t * matrix_src = (ধ্রুবক uint8_t *) t->তথ্য + slice_idx * matrix_আকার;

        জন্য (পূর্ণসংখ্যা ct = start_ct; ct < end_ct; ct++) {
            জন্য (পূর্ণসংখ্যা kt = 0; kt < n_k_tiles; kt++) {
                ধ্রুবক uint8_t * tile_src = matrix_src + ((size_t) ct * n_k_tiles + kt) * tile_আকার;
                unpack_tile(dst_slice, tile_src, ne0, start_row, end_row, row_stride, ct, kt);
            }
        }
    }
}

// --- Block-32 Quantized Types (Q4_0, IQ4_NL, Q4_1, Q8_0, MXFP4) ---

স্থির শূন্য repack_q4_0_tiled(ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t offset, size_t size) {
    repack_to_tiled_grid_32<block_q4_0, 1>(t, তথ্য, offset, size, HTP_TENSOR_TILE_আকার_Q4_0,
        [](uint8_t * tile_dst, ধ্রুবক block_q4_0 * src_slice, int64_t ne0, int64_t ne1, int64_t row_stride, পূর্ণসংখ্যা ct, পূর্ণসংখ্যা kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
#যদি defined(__ARM_NEON) || defined(__aarch64__)
            যদি (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                জন্য (পূর্ণসংখ্যা row0 = 0; row0 < 32; row0 += 8) {
                    ধ্রুবক uint8_t * qs[8];
                    জন্য (পূর্ণসংখ্যা rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_slice[(ct * 32 + row0 + rr) * row_stride + kt].qs;
                    }
                    repack_neon_4bit_rows(qs, tile_dst, row0);
                }

                জন্য (পূর্ণসংখ্যা row = 0; row < 32; ++row) {
                    scale_dst[row] = src_slice[(ct * 32 + row) * row_stride + kt].d;
                }
                ফেরত;
            }
#endif
            uint8_t tile_quants[32][32];
            জন্য (পূর্ণসংখ্যা row = 0; row < 32; row++) {
                ধ্রুবক int64_t r = ct * 32 + row;
                যদি (r < ne1 && kt < ne0 / 32) {
                    ধ্রুবক block_q4_0 & b = src_slice[r * row_stride + kt];
                    unpack_nibbles_32(tile_quants[row], b.qs);
                    scale_dst[row] = b.d;
                } অন্যথা {
                    memset(tile_quants[row], 8, 32);
                    scale_dst[row] = 0;
                }
            }
            transpose_nibbles_to_tile(tile_dst, tile_quants);
        });
}

স্থির শূন্য repack_tiled_q4_0(শূন্য * তথ্য, ধ্রুবক ggml_tensor * t, size_t offset, size_t size) {
    unpack_from_tiled_grid_32<block_q4_0, 1>(তথ্য, t, offset, size, HTP_TENSOR_TILE_আকার_Q4_0,
        [](block_q4_0 * dst_slice, ধ্রুবক uint8_t * tile_src, int64_t ne0, int64_t start_row, int64_t end_row, int64_t row_stride, পূর্ণসংখ্যা ct, পূর্ণসংখ্যা kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_to_nibbles(tile_quants, tile_src);
            ধ্রুবক ggml_half * scale_src = (ধ্রুবক ggml_half *) (tile_src + 512);
            জন্য (পূর্ণসংখ্যা row = 0; row < 32; row++) {
                ধ্রুবক int64_t r = ct * 32 + row;
                যদি (r >= start_row && r < end_row && kt < ne0 / 32) {
                    block_q4_0 & b = dst_slice[(r - start_row) * row_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[row]);
                    b.d = scale_src[row];
                }
            }
        });
}

স্থির শূন্য repack_q4_1_tiled(ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t offset, size_t size) {
    repack_to_tiled_grid_32<block_q4_1, 1>(t, তথ্য, offset, size, HTP_TENSOR_TILE_আকার_Q4_1,
        [](uint8_t * tile_dst, ধ্রুবক block_q4_1 * src_slice, int64_t ne0, int64_t ne1, int64_t row_stride, পূর্ণসংখ্যা ct, পূর্ণসংখ্যা kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
#যদি defined(__ARM_NEON) || defined(__aarch64__)
            যদি (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                জন্য (পূর্ণসংখ্যা row0 = 0; row0 < 32; row0 += 8) {
                    ধ্রুবক uint8_t * qs[8];
                    জন্য (পূর্ণসংখ্যা rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_slice[(ct * 32 + row0 + rr) * row_stride + kt].qs;
                    }
                    repack_neon_4bit_rows(qs, tile_dst, row0);
                }

                জন্য (পূর্ণসংখ্যা row = 0; row < 32; ++row) {
                    ধ্রুবক block_q4_1 & src_block = src_slice[(ct * 32 + row) * row_stride + kt];
                    scale_dst[2 * row + 0] = src_block.d;
                    scale_dst[2 * row + 1] = src_block.m;
                }
                ফেরত;
            }
#endif
            uint8_t tile_quants[32][32];
            জন্য (পূর্ণসংখ্যা row = 0; row < 32; row++) {
                ধ্রুবক int64_t r = ct * 32 + row;
                যদি (r < ne1 && kt < ne0 / 32) {
                    ধ্রুবক block_q4_1 & b = src_slice[r * row_stride + kt];
                    unpack_nibbles_32(tile_quants[row], b.qs);
                    scale_dst[2 * row + 0] = b.d;
                    scale_dst[2 * row + 1] = b.m;
                } অন্যথা {
                    memset(tile_quants[row], 0, 32);
                    scale_dst[2 * row + 0] = 0;
                    scale_dst[2 * row + 1] = 0;
                }
            }
            transpose_nibbles_to_tile(tile_dst, tile_quants);
        });
}

স্থির শূন্য repack_tiled_q4_1(শূন্য * তথ্য, ধ্রুবক ggml_tensor * t, size_t offset, size_t size) {
    unpack_from_tiled_grid_32<block_q4_1, 1>(তথ্য, t, offset, size, HTP_TENSOR_TILE_আকার_Q4_1,
        [](block_q4_1 * dst_slice, ধ্রুবক uint8_t * tile_src, int64_t ne0, int64_t start_row, int64_t end_row, int64_t row_stride, পূর্ণসংখ্যা ct, পূর্ণসংখ্যা kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_to_nibbles(tile_quants, tile_src);
            ধ্রুবক ggml_half * scale_src = (ধ্রুবক ggml_half *) (tile_src + 512);
            জন্য (পূর্ণসংখ্যা row = 0; row < 32; row++) {
                ধ্রুবক int64_t r = ct * 32 + row;
                যদি (r >= start_row && r < end_row && kt < ne0 / 32) {
                    block_q4_1 & b = dst_slice[(r - start_row) * row_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[row]);
                    b.d = scale_src[2 * row + 0];
                    b.m = scale_src[2 * row + 1];
                }
            }
        });
}

স্থির শূন্য repack_mxfp4_tiled(ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t offset, size_t size) {
    repack_to_tiled_grid_32<block_mxfp4, 1>(t, তথ্য, offset, size, HTP_TENSOR_TILE_আকার_MXFP4,
        [](uint8_t * tile_dst, ধ্রুবক block_mxfp4 * src_slice, int64_t ne0, int64_t ne1, int64_t row_stride, পূর্ণসংখ্যা ct, পূর্ণসংখ্যা kt) {
            uint8_t * scale_dst = tile_dst + 512;
#যদি defined(__ARM_NEON) || defined(__aarch64__)
            যদি (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                জন্য (পূর্ণসংখ্যা row0 = 0; row0 < 32; row0 += 8) {
                    ধ্রুবক uint8_t * qs[8];
                    জন্য (পূর্ণসংখ্যা rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_slice[(ct * 32 + row0 + rr) * row_stride + kt].qs;
                    }
                    repack_neon_4bit_rows(qs, tile_dst, row0);
                }

                জন্য (পূর্ণসংখ্যা row = 0; row < 32; ++row) {
                    scale_dst[row] = src_slice[(ct * 32 + row) * row_stride + kt].e;
                }
                ফেরত;
            }
#endif
            uint8_t tile_quants[32][32];
            জন্য (পূর্ণসংখ্যা row = 0; row < 32; row++) {
                ধ্রুবক int64_t r = ct * 32 + row;
                যদি (r < ne1 && kt < ne0 / 32) {
                    ধ্রুবক block_mxfp4 & b = src_slice[r * row_stride + kt];
                    unpack_nibbles_32(tile_quants[row], b.qs);
                    scale_dst[row] = b.e;
                } অন্যথা {
                    memset(tile_quants[row], 0, 32);
                    scale_dst[row] = 0;
                }
            }
            transpose_nibbles_to_tile(tile_dst, tile_quants);
        });
}

স্থির শূন্য repack_tiled_mxfp4(শূন্য * তথ্য, ধ্রুবক ggml_tensor * t, size_t offset, size_t size) {
    unpack_from_tiled_grid_32<block_mxfp4, 1>(তথ্য, t, offset, size, HTP_TENSOR_TILE_আকার_MXFP4,
        [](block_mxfp4 * dst_slice, ধ্রুবক uint8_t * tile_src, int64_t ne0, int64_t start_row, int64_t end_row, int64_t row_stride, পূর্ণসংখ্যা ct, পূর্ণসংখ্যা kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_to_nibbles(tile_quants, tile_src);
            ধ্রুবক uint8_t * scale_src = tile_src + 512;
            জন্য (পূর্ণসংখ্যা row = 0; row < 32; row++) {
                ধ্রুবক int64_t r = ct * 32 + row;
                যদি (r >= start_row && r < end_row && kt < ne0 / 32) {
                    block_mxfp4 & b = dst_slice[(r - start_row) * row_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[row]);
                    b.e = scale_src[row];
                }
            }
        });
}

স্থির শূন্য repack_q8_0_tiled(ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t offset, size_t size) {
    repack_to_tiled_grid_32<block_q8_0, 1>(t, তথ্য, offset, size, HTP_TENSOR_TILE_আকার_Q8_0,
        [](uint8_t * tile_dst, ধ্রুবক block_q8_0 * src_slice, int64_t ne0, int64_t ne1, int64_t row_stride, পূর্ণসংখ্যা ct, পূর্ণসংখ্যা kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 1024);
#যদি defined(__ARM_NEON) || defined(__aarch64__)
            যদি (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                জন্য (পূর্ণসংখ্যা row0 = 0; row0 < 32; row0 += 8) {
                    ধ্রুবক uint8_t * qs[8];
                    জন্য (পূর্ণসংখ্যা rr = 0; rr < 8; ++rr) {
                        qs[rr] = (ধ্রুবক uint8_t *) src_slice[(ct * 32 + row0 + rr) * row_stride + kt].qs;
                    }
                    repack_neon_q8_rows(qs, tile_dst, row0);
                }

                জন্য (পূর্ণসংখ্যা row = 0; row < 32; ++row) {
                    scale_dst[row] = src_slice[(ct * 32 + row) * row_stride + kt].d;
                }
                ফেরত;
            }
#endif
            জন্য (পূর্ণসংখ্যা cp = 0; cp < 16; cp++) {
                ধ্রুবক পূর্ণসংখ্যা col0 = cp * 2;
                ধ্রুবক পূর্ণসংখ্যা col1 = col0 + 1;
                জন্য (পূর্ণসংখ্যা row = 0; row < 32; row++) {
                    ধ্রুবক int64_t r = ct * 32 + row;
                    ধ্রুবক block_q8_0 * b = (r < ne1 && kt < ne0 / 32) ? &src_slice[r * row_stride + kt] : nullptr;
                    tile_dst[cp * 64 + 2 * row + 0] = b ? b->qs[col0] : 0;
                    tile_dst[cp * 64 + 2 * row + 1] = b ? b->qs[col1] : 0;
                }
            }
            জন্য (পূর্ণসংখ্যা row = 0; row < 32; row++) {
                ধ্রুবক int64_t r = ct * 32 + row;
                scale_dst[row] = (r < ne1 && kt < ne0 / 32) ? src_slice[r * row_stride + kt].d : 0;
            }
        });
}

স্থির শূন্য repack_tiled_q8_0(শূন্য * তথ্য, ধ্রুবক ggml_tensor * t, size_t offset, size_t size) {
    unpack_from_tiled_grid_32<block_q8_0, 1>(তথ্য, t, offset, size, HTP_TENSOR_TILE_আকার_Q8_0,
        [](block_q8_0 * dst_slice, ধ্রুবক uint8_t * tile_src, int64_t ne0, int64_t start_row, int64_t end_row, int64_t row_stride, পূর্ণসংখ্যা ct, পূর্ণসংখ্যা kt) {
            জন্য (পূর্ণসংখ্যা cp = 0; cp < 16; cp++) {
                ধ্রুবক পূর্ণসংখ্যা col0 = cp * 2;
                ধ্রুবক পূর্ণসংখ্যা col1 = col0 + 1;
                জন্য (পূর্ণসংখ্যা row = 0; row < 32; row++) {
                    ধ্রুবক int64_t r = ct * 32 + row;
                    যদি (r >= start_row && r < end_row && kt < ne0 / 32) {
                        block_q8_0 & b = dst_slice[(r - start_row) * row_stride + kt];
                        b.qs[col0] = tile_src[cp * 64 + 2 * row + 0];
                        b.qs[col1] = tile_src[cp * 64 + 2 * row + 1];
                    }
                }
            }
            ধ্রুবক ggml_half * scale_src = (ধ্রুবক ggml_half *) (tile_src + 1024);
            জন্য (পূর্ণসংখ্যা row = 0; row < 32; row++) {
                ধ্রুবক int64_t r = ct * 32 + row;
                যদি (r >= start_row && r < end_row && kt < ne0 / 32) {
                    dst_slice[(r - start_row) * row_stride + kt].d = scale_src[row];
                }
            }
        });
}

// --- Float Types (F16, F32, BF16 -> f16_tiled HMX layout) ---

স্থির শূন্য repack_q2_0_tiled(ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(t->ne[0] % 64 == 0);
    GGML_ASSERT(size == ggml_nbytes(t));

    ধ্রুবক block_q2_0 * src = (ধ্রুবক block_q2_0 *) তথ্য;
    ধ্রুবক int64_t blocks_per_row = t->ne[0] / 64;
    ধ্রুবক int64_t n_col_tiles = hex_round_up(t->ne[1], 32) / 32;
    ধ্রুবক int64_t n_weight_tiles = blocks_per_row;
    ধ্রুবক size_t matrix_আকার = (size_t) n_col_tiles * n_weight_tiles * HTP_TENSOR_TILE_আকার_Q2_0;

    জন্য (int64_t slice = 0; slice < t->ne[2] * t->ne[3]; ++slice) {
        ধ্রুবক block_q2_0 * src_slice = src + slice * t->ne[1] * blocks_per_row;
        uint8_t * dst_slice = (uint8_t *) t->তথ্য + slice * matrix_আকার;
        ধ্রুবক size_t ct_bytes = (size_t) n_weight_tiles * HTP_TENSOR_TILE_আকার_Q2_0;

        repack_parallel_for_ct(matrix_আকার, (পূর্ণসংখ্যা) n_col_tiles, [&](পূর্ণসংখ্যা ct0, পূর্ণসংখ্যা ct1) {
            memset(dst_slice + (size_t) ct0 * ct_bytes, 0, (size_t) (ct1 - ct0) * ct_bytes);
            ধ্রুবক int64_t r0 = (int64_t) ct0 * 32;
            ধ্রুবক int64_t r1 = (std::min)((int64_t) ct1 * 32, t->ne[1]);
            জন্য (int64_t row_সূচক = r0; row_সূচক < r1; ++row_সূচক) {
                ধ্রুবক int64_t ct = row_সূচক / 32;
                ধ্রুবক পূর্ণসংখ্যা row = (পূর্ণসংখ্যা) (row_সূচক % 32);
                জন্য (int64_t kt = 0; kt < n_weight_tiles; ++kt) {
                    ধ্রুবক block_q2_0 * block = src_slice + row_সূচক * blocks_per_row + kt;
                    uint8_t * tile = dst_slice + (ct * n_weight_tiles + kt) * HTP_TENSOR_TILE_আকার_Q2_0;
#যদি defined(__ARM_NEON) || defined(__aarch64__)
                    repack_neon_q2_0_block(block->qs, tile, row);
#অন্যথা
                    জন্য (পূর্ণসংখ্যা group = 0; group < 16; ++group) {
                        জন্য (পূর্ণসংখ্যা byte = 0; byte < 4; ++byte) {
                            ধ্রুবক পূর্ণসংখ্যা k = 4 * group + byte;
                            ধ্রুবক uint8_t code = (block->qs[k / 4] >> (2 * (k % 4))) & 3;
                            tile[(group >> 2) * 128 + 4 * row + byte] |= code << (2 * (group & 3));
                        }
                    }
#endif
                    memcpy(tile + 512 + 2 * row, &block->d, আকার(block->d));
                }
            }
        });
    }
}

স্থির শূন্য repack_tiled_q2_0(শূন্য * তথ্য, ধ্রুবক ggml_tensor * t, size_t offset, size_t size) {
    GGML_ASSERT(t->ne[0] % 64 == 0);
    GGML_ASSERT(offset + size <= ggml_nbytes(t));

    ধ্রুবক size_t first_block = offset / আকার(block_q2_0);
    ধ্রুবক size_t end_block = (offset + size + আকার(block_q2_0) - 1) / আকার(block_q2_0);
    ধ্রুবক size_t blocks_per_row = t->ne[0] / 64;
    ধ্রুবক size_t blocks_per_slice = t->ne[1] * blocks_per_row;
    ধ্রুবক size_t n_weight_tiles = blocks_per_row;
    ধ্রুবক size_t n_col_tiles = hex_round_up(t->ne[1], 32) / 32;
    ধ্রুবক size_t matrix_আকার = n_col_tiles * n_weight_tiles * HTP_TENSOR_TILE_আকার_Q2_0;

    জন্য (size_t block_সূচক = first_block; block_সূচক < end_block; ++block_সূচক) {
        ধ্রুবক size_t slice = block_সূচক / blocks_per_slice;
        ধ্রুবক size_t in_slice = block_সূচক % blocks_per_slice;
        ধ্রুবক size_t row_সূচক = in_slice / blocks_per_row;
        ধ্রুবক size_t kt = in_slice % blocks_per_row;
        ধ্রুবক size_t ct = row_সূচক / 32;
        ধ্রুবক size_t row = row_সূচক % 32;
        ধ্রুবক uint8_t * tile = (ধ্রুবক uint8_t *) t->তথ্য + slice * matrix_আকার +
            (ct * n_weight_tiles + kt) * HTP_TENSOR_TILE_আকার_Q2_0;

        block_q2_0 block = {};
        memcpy(&block.d, tile + 512 + 2 * row, আকার(block.d));
        জন্য (পূর্ণসংখ্যা group = 0; group < 16; ++group) {
            জন্য (পূর্ণসংখ্যা byte = 0; byte < 4; ++byte) {
                ধ্রুবক uint8_t packed = tile[(group >> 2) * 128 + 4 * row + byte];
                ধ্রুবক uint8_t code = (packed >> (2 * (group & 3))) & 3;
                block.qs[group] |= code << (2 * byte);
            }
        }

        ধ্রুবক size_t block_begin = block_সূচক * আকার(block_q2_0);
        ধ্রুবক size_t copy_begin = (std::max)(block_begin, offset);
        ধ্রুবক size_t copy_end = (std::min)(block_begin + আকার(block_q2_0), offset + size);
        memcpy((uint8_t *) তথ্য + copy_begin - offset, (ধ্রুবক uint8_t *) &block + copy_begin - block_begin, copy_end - copy_begin);
    }
}

template <typename SrcT, typename ToFp16Fn>
স্থির শূন্য repack_float_to_f16_tiled(ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t offset, size_t size, ToFp16Fn && to_fp16) {
    repack_to_tiled_grid_32<SrcT, 32>(t, তথ্য, offset, size, HTP_TENSOR_TILE_আকার_F16,
        [&](uint8_t * tile_bytes, ধ্রুবক SrcT * src_slice, int64_t ne0, int64_t ne1, int64_t row_stride, পূর্ণসংখ্যা ct, পূর্ণসংখ্যা kt) {
            uint16_t * tile_dst = (uint16_t *) tile_bytes;
#যদি defined(__ARM_NEON) || defined(__aarch64__)
            যদি (ct * 32 + 32 <= ne1 && kt * 32 + 32 <= ne0) {
                repack_neon_f16_tile(tile_dst, src_slice, row_stride, ct, kt);
                ফেরত;
            }
#endif
            জন্য (পূর্ণসংখ্যা cp = 0; cp < 16; cp++) {
                ধ্রুবক int64_t c0 = kt * 32 + cp * 2;
                ধ্রুবক int64_t c1 = c0 + 1;
                জন্য (পূর্ণসংখ্যা row = 0; row < 32; row++) {
                    ধ্রুবক int64_t r = ct * 32 + row;
                    ধ্রুবক SrcT * r_ptr = (r < ne1) ? &src_slice[r * row_stride] : nullptr;
                    tile_dst[cp * 64 + row * 2 + 0] = (r_ptr && c0 < ne0) ? to_fp16(r_ptr[c0]) : 0;
                    tile_dst[cp * 64 + row * 2 + 1] = (r_ptr && c1 < ne0) ? to_fp16(r_ptr[c1]) : 0;
                }
            }
        });
}

template <typename DstT, typename FromFp16Fn>
স্থির শূন্য unpack_f16_tiled_to_float(শূন্য * তথ্য, ধ্রুবক ggml_tensor * t, size_t offset, size_t size, FromFp16Fn && from_fp16) {
    unpack_from_tiled_grid_32<DstT, 32>(তথ্য, t, offset, size, HTP_TENSOR_TILE_আকার_F16,
        [&](DstT * dst_slice, ধ্রুবক uint8_t * tile_bytes, int64_t ne0, int64_t start_row, int64_t end_row, int64_t row_stride, পূর্ণসংখ্যা ct, পূর্ণসংখ্যা kt) {
            ধ্রুবক uint16_t * tile_src = (ধ্রুবক uint16_t *) tile_bytes;
            জন্য (পূর্ণসংখ্যা cp = 0; cp < 16; cp++) {
                ধ্রুবক int64_t c0 = kt * 32 + cp * 2;
                ধ্রুবক int64_t c1 = c0 + 1;
                জন্য (পূর্ণসংখ্যা row = 0; row < 32; row++) {
                    ধ্রুবক int64_t r = ct * 32 + row;
                    যদি (r >= start_row && r < end_row) {
                        DstT * r_ptr = dst_slice + (r - start_row) * row_stride;
                        যদি (c0 < ne0) r_ptr[c0] = from_fp16(tile_src[cp * 64 + row * 2 + 0]);
                        যদি (c1 < ne0) r_ptr[c1] = from_fp16(tile_src[cp * 64 + row * 2 + 1]);
                    }
                }
            }
        });
}

// --- Generic K-Quant (super-block 256) grid walkers ---

template <typename BlockT, typename PackKtFn>
স্থির শূন্য repack_kquant_to_tiled_grid(ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t offset, size_t size,
                                        size_t tile_আকার, PackKtFn && pack_kt) {
    GGML_ASSERT(offset == 0);
    GGML_UNUSED(size);

    ধ্রুবক BlockT * src_matrix = (ধ্রুবক BlockT *) তথ্য;
    ধ্রুবক int64_t ne0 = t->ne[0];
    ধ্রুবক int64_t ne1 = t->ne[1];
    ধ্রুবক int64_t ne2 = t->ne[2];
    ধ্রুবক int64_t ne3 = t->ne[3];
    GGML_ASSERT(ne0 % QK_K == 0);

    ধ্রুবক পূর্ণসংখ্যা n_col_tiles    = hex_round_up(ne1, 32) / 32;
    ধ্রুবক পূর্ণসংখ্যা n_k_tiles      = hex_round_up(ne0, 32) / 32;
    ধ্রুবক size_t matrix_আকার = (size_t) n_col_tiles * n_k_tiles * tile_আকার;
    ধ্রুবক int64_t sb_per_row = ne0 / QK_K;

    জন্য (int64_t slice_idx = 0; slice_idx < ne2 * ne3; slice_idx++) {
        ধ্রুবক BlockT * src_slice = src_matrix + slice_idx * (ne1 * sb_per_row);
        uint8_t * matrix_dst     = (uint8_t *) t->তথ্য + slice_idx * matrix_আকার;
        ধ্রুবক size_t ct_bytes    = (size_t) n_k_tiles * tile_আকার;

        repack_parallel_for_ct(matrix_আকার, n_col_tiles, [&](পূর্ণসংখ্যা ct0, পূর্ণসংখ্যা ct1) {
            // Padding rows and OR-ed bit planes require zeroed tiles
            memset(matrix_dst + (size_t) ct0 * ct_bytes, 0, (size_t) (ct1 - ct0) * ct_bytes);

            ধ্রুবক int64_t r0 = (int64_t) ct0 * 32;
            ধ্রুবক int64_t r1 = (std::min)((int64_t) ct1 * 32, ne1);
            জন্য (int64_t r = r0; r < r1; r++) {
                ধ্রুবক পূর্ণসংখ্যা ct  = (পূর্ণসংখ্যা) (r / 32);
                ধ্রুবক পূর্ণসংখ্যা row = (পূর্ণসংখ্যা) (r % 32);
                ধ্রুবক BlockT * src_row = src_slice + r * sb_per_row;

                জন্য (পূর্ণসংখ্যা kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile = matrix_dst + ((size_t) ct * n_k_tiles + kt) * tile_আকার;
                    pack_kt(tile, &src_row[kt / 8], row, kt % 8);
                }
            }
        });
    }
}

template <typename BlockT, typename UnpackSbFn>
স্থির শূন্য unpack_kquant_from_tiled_grid(শূন্য * তথ্য, ধ্রুবক ggml_tensor * t, size_t offset, size_t size,
                                          size_t tile_আকার, UnpackSbFn && unpack_sb) {
    uint8_t * dst_bytes = (uint8_t *) তথ্য;
    ধ্রুবক int64_t ne0 = t->ne[0];
    ধ্রুবক int64_t ne1 = t->ne[1];
    ধ্রুবক int64_t ne2 = t->ne[2];
    ধ্রুবক int64_t ne3 = t->ne[3];
    GGML_ASSERT(ne0 % QK_K == 0);

    ধ্রুবক পূর্ণসংখ্যা n_col_tiles    = hex_round_up(ne1, 32) / 32;
    ধ্রুবক পূর্ণসংখ্যা n_k_tiles      = hex_round_up(ne0, 32) / 32;
    ধ্রুবক size_t matrix_আকার = (size_t) n_col_tiles * n_k_tiles * tile_আকার;
    ধ্রুবক int64_t sb_per_row = ne0 / QK_K;

    ধ্রুবক size_t row_আকার_bytes = ggml_row_আকার(t->type, ne0);
    ধ্রুবক size_t slice_আকার     = (size_t) ne1 * row_আকার_bytes;
    ধ্রুবক int64_t start_slice   = offset / slice_আকার;
    int64_t end_slice           = (offset + size + slice_আকার - 1) / slice_আকার;
    যদি (end_slice > ne2 * ne3) {
        end_slice = ne2 * ne3;
    }

    জন্য (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        ধ্রুবক size_t cur_start_byte     = (std::max)(offset, (size_t) slice_idx * slice_আকার);
        ধ্রুবক size_t cur_end_byte       = (std::min)(offset + size, (size_t) (slice_idx + 1) * slice_আকার);
        ধ্রুবক size_t slice_offset_start = cur_start_byte - (size_t) slice_idx * slice_আকার;
        ধ্রুবক size_t slice_offset_end   = cur_end_byte - (size_t) slice_idx * slice_আকার;

        ধ্রুবক int64_t start_row = slice_offset_start / row_আকার_bytes;
        ধ্রুবক int64_t end_row   = (std::min)((int64_t) ((slice_offset_end + row_আকার_bytes - 1) / row_আকার_bytes), ne1);

        BlockT * dst_slice         = (BlockT *) (dst_bytes + (cur_start_byte - offset));
        ধ্রুবক uint8_t * matrix_src = (ধ্রুবক uint8_t *) t->তথ্য + slice_idx * matrix_আকার;

        জন্য (int64_t r = start_row; r < end_row; r++) {
            ধ্রুবক পূর্ণসংখ্যা ct  = (পূর্ণসংখ্যা) (r / 32);
            ধ্রুবক পূর্ণসংখ্যা row = (পূর্ণসংখ্যা) (r % 32);
            BlockT * dst_row = dst_slice + (r - start_row) * sb_per_row;

            জন্য (int64_t sb = 0; sb < sb_per_row; sb++) {
                BlockT * b = &dst_row[sb];
                memset(b, 0, আকার(BlockT));
                ধ্রুবক uint8_t * sb_tiles = matrix_src + ((size_t) ct * n_k_tiles + sb * 8) * tile_আকার;
                unpack_sb(b, sb_tiles, tile_আকার, row);
            }
        }
    }
}

// --- K-Quant Helpers & Per-Type Implementations (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) ---

#যদি !defined(__ARM_NEON) && !defined(__aarch64__)
// unsigned 6-bit value (0..63) of element e of a Q6_K block, same bit layout as dequantize_row_q6_K
স্থির অন্তঃরেখ uint8_t q6_K_get_quant(ধ্রুবক block_q6_K * b, পূর্ণসংখ্যা e) {
    ধ্রুবক পূর্ণসংখ্যা c = e / 128;
    ধ্রুবক পূর্ণসংখ্যা w = e % 128;
    ধ্রুবক পূর্ণসংখ্যা g = w / 32;
    ধ্রুবক পূর্ণসংখ্যা l = w % 32;
    ধ্রুবক uint8_t * ql = b->ql + c * 64;
    ধ্রুবক uint8_t * qh = b->qh + c * 32;
    uint8_t lo, hi;
    নির্বাচন (g) {
        ক্ষেত্র 0:  lo = ql[l]      & 0xF; hi = (qh[l] >> 0) & 3; বিরতি;
        ক্ষেত্র 1:  lo = ql[l + 32] & 0xF; hi = (qh[l] >> 2) & 3; বিরতি;
        ক্ষেত্র 2:  lo = ql[l]      >> 4;  hi = (qh[l] >> 4) & 3; বিরতি;
        পূর্বনির্ধারিত: lo = ql[l + 32] >> 4;  hi = (qh[l] >> 6) & 3; বিরতি;
    }
    ফেরত (uint8_t) (lo | (hi << 4));
}
#endif

স্থির শূন্য repack_q6_K_tiled(ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t offset, size_t size) {
    repack_kquant_to_tiled_grid<block_q6_K>(t, তথ্য, offset, size, HTP_TENSOR_TILE_আকার_Q6_K,
        [](uint8_t * tile, ধ্রুবক block_q6_K * b, পূর্ণসংখ্যা row, পূর্ণসংখ্যা kt_local) {
            ধ্রুবক দশমিক d     = GGML_FP16_TO_FP32(b->d);
            uint8_t * lo_pl   = tile;
            uint8_t * hi_pl   = tile + 512;
            ggml_half * sc_pl = (ggml_half *) (tile + 768);

#যদি defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q6_k(b, kt_local, lo_pl, hi_pl, row);
#অন্যথা
            জন্য (পূর্ণসংখ্যা lk = 0; lk < 32; lk++) {
                ধ্রুবক uint8_t q6 = q6_K_get_quant(b, kt_local * 32 + lk);
                ধ্রুবক পূর্ণসংখ্যা g   = lk >> 2;
                ধ্রুবক পূর্ণসংখ্যা pos = row * 4 + (lk & 3);
                lo_pl[(g >> 1) * 128 + pos] |= (uint8_t) ((q6 & 0xF) << ((g & 1) * 4));
                hi_pl[(g >> 2) * 128 + pos] |= (uint8_t) ((q6 >> 4) << ((g & 3) * 2));
            }
#endif
            জন্য (পূর্ণসংখ্যা sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + row] = GGML_FP32_TO_FP16(d * (দশমিক) b->scales[kt_local * 2 + sub]);
            }
        });
}

স্থির শূন্য repack_tiled_q6_K(শূন্য * তথ্য, ধ্রুবক ggml_tensor * t, size_t offset, size_t size) {
    unpack_kquant_from_tiled_grid<block_q6_K>(তথ্য, t, offset, size, HTP_TENSOR_TILE_আকার_Q6_K,
        [](block_q6_K * b, ধ্রুবক uint8_t * sb_tiles, size_t tile_আকার, পূর্ণসংখ্যা row) {
            দশমিক sub_scales[16];
            জন্য (পূর্ণসংখ্যা kt_local = 0; kt_local < 8; kt_local++) {
                ধ্রুবক uint8_t *   tile  = sb_tiles + kt_local * tile_আকার;
                ধ্রুবক uint8_t *   lo_pl = tile;
                ধ্রুবক uint8_t *   hi_pl = tile + 512;
                ধ্রুবক ggml_half * sc_pl = (ধ্রুবক ggml_half *) (tile + 768);

                ধ্রুবক পূর্ণসংখ্যা c  = kt_local / 4;
                ধ্রুবক পূর্ণসংখ্যা g  = kt_local % 4;
                uint8_t * ql = b->ql + c * 64;
                uint8_t * qh = b->qh + c * 32;

                জন্য (পূর্ণসংখ্যা lk = 0; lk < 32; lk++) {
                    ধ্রুবক পূর্ণসংখ্যা g_tile = lk >> 2;
                    ধ্রুবক পূর্ণসংখ্যা pos    = row * 4 + (lk & 3);
                    ধ্রুবক uint8_t lo = (lo_pl[(g_tile >> 1) * 128 + pos] >> ((g_tile & 1) * 4)) & 0xF;
                    ধ্রুবক uint8_t hi = (hi_pl[(g_tile >> 2) * 128 + pos] >> ((g_tile & 3) * 2)) & 3;

                    নির্বাচন (g) {
                        ক্ষেত্র 0:  ql[lk]      |= lo;        qh[lk] |= (hi << 0); বিরতি;
                        ক্ষেত্র 1:  ql[lk + 32] |= lo;        qh[lk] |= (hi << 2); বিরতি;
                        ক্ষেত্র 2:  ql[lk]      |= (lo << 4); qh[lk] |= (hi << 4); বিরতি;
                        পূর্বনির্ধারিত: ql[lk + 32] |= (lo << 4); qh[lk] |= (hi << 6); বিরতি;
                    }
                }

                জন্য (পূর্ণসংখ্যা sub = 0; sub < 2; sub++) {
                    sub_scales[kt_local * 2 + sub] = GGML_FP16_TO_FP32(sc_pl[sub * 32 + row]);
                }
            }

            দশমিক max_abs_scale = 0.0f;
            জন্য (পূর্ণসংখ্যা s = 0; s < 16; s++) {
                max_abs_scale = (std::max)(max_abs_scale, fabsf(sub_scales[s]));
            }

            যদি (max_abs_scale == 0.0f) {
                b->d = GGML_FP32_TO_FP16(0.0f);
            } অন্যথা {
                b->d = GGML_FP32_TO_FP16(max_abs_scale / 127.0f);
                ধ্রুবক দশমিক d_actual = GGML_FP16_TO_FP32(b->d);
                ধ্রুবক দশমিক inv_d    = (d_actual != 0.0f) ? (1.0f / d_actual) : 0.0f;
                জন্য (পূর্ণসংখ্যা s = 0; s < 16; s++) {
                    ধ্রুবক পূর্ণসংখ্যা sc = (পূর্ণসংখ্যা) roundf(sub_scales[s] * inv_d);
                    b->scales[s] = (int8_t) (std::max)(-128, (std::min)(127, sc));
                }
            }
        });
}

#যদি !defined(__ARM_NEON) && !defined(__aarch64__)
// low 2 bits (0..3) of element e of a Q2_K or Q3_K block, same bit layout as dequantize_row_q2_K / q3_K
স্থির অন্তঃরেখ uint8_t q2_3_K_get_low2(ধ্রুবক uint8_t * qs, পূর্ণসংখ্যা e) {
    ধ্রুবক পূর্ণসংখ্যা c = e / 128;
    ধ্রুবক পূর্ণসংখ্যা j = (e % 128) / 32;
    ধ্রুবক পূর্ণসংখ্যা l = e % 32;
    ফেরত (qs[c * 32 + l] >> (2 * j)) & 3;
}

// hmask bit of element e of a Q3_K block, same bit layout as dequantize_row_q3_K
স্থির অন্তঃরেখ যৌক্তিক q3_K_get_hbit(ধ্রুবক block_q3_K * b, পূর্ণসংখ্যা e) {
    ধ্রুবক পূর্ণসংখ্যা c = e / 128;
    ধ্রুবক পূর্ণসংখ্যা j = (e % 128) / 32;
    ধ্রুবক পূর্ণসংখ্যা l = e % 32;
    ফেরত (b->hmask[l] >> (c * 4 + j)) & 1;
}
#endif

// signed 6-bit scale j (-32..31) of a Q3_K block, same packing as quantize_row_q3_K_ref
স্থির অন্তঃরেখ পূর্ণসংখ্যা q3_K_get_scale(ধ্রুবক uint8_t * scales, পূর্ণসংখ্যা j) {
    ধ্রুবক পূর্ণসংখ্যা lo = (j < 8) ? (scales[j] & 0xF) : (scales[j - 8] >> 4);
    ধ্রুবক পূর্ণসংখ্যা hi = (scales[8 + j % 4] >> (2 * (j / 4))) & 3;
    ফেরত (lo | (hi << 4)) - 32;
}

// read-back: find fp16 d and l[j] in [lmin, lmax] with fp16(d * l[j]) == prod[j] for all j, false if none
স্থির যৌক্তিক hexagon_recover_k_scales(ধ্রুবক ggml_half * prod, পূর্ণসংখ্যা n, পূর্ণসংখ্যা lmin, পূর্ণসংখ্যা lmax, ggml_half * d_out, পূর্ণসংখ্যা * l_out) {
    পূর্ণসংখ্যা jmax = 0;
    জন্য (পূর্ণসংখ্যা j = 1; j < n; j++) {
        যদি (fabsf(GGML_FP16_TO_FP32(prod[j])) > fabsf(GGML_FP16_TO_FP32(prod[jmax]))) {
            jmax = j;
        }
    }
    ধ্রুবক দশমিক pmax = GGML_FP16_TO_FP32(prod[jmax]);
    যদি (pmax == 0.0f) {
        *d_out = GGML_FP32_TO_FP16(0.0f);
        জন্য (পূর্ণসংখ্যা j = 0; j < n; j++) {
            l_out[j] = 0;
        }
        ফেরত সত্য;
    }

    // the quantizers put the largest scale at or near the range end, so try large |l| first
    ধ্রুবক পূর্ণসংখ্যা lext = (std::max)(-lmin, lmax);
    জন্য (পূর্ণসংখ্যা a = lext; a >= 1; a--) {
        জন্য (পূর্ণসংখ্যা sign : { 1, -1 }) {
            ধ্রুবক পূর্ণসংখ্যা lj = sign * a;
            যদি (lj < lmin || lj > lmax) {
                চালু;
            }
            ধ্রুবক ggml_half d0 = GGML_FP32_TO_FP16(pmax / (দশমিক) lj);
            জন্য (পূর্ণসংখ্যা ulp : { 0, -1, 1 }) {
                ggml_half d = d0;
                uint16_t bits;
                memcpy(&bits, &d, আকার(bits));
                bits = (uint16_t) (bits + ulp);
                memcpy(&d, &bits, আকার(bits));

                ধ্রুবক দশমিক df = GGML_FP16_TO_FP32(d);
                যদি (!std::isfinite(df) || df == 0.0f) {
                    চালু;
                }
                যৌক্তিক ok = সত্য;
                জন্য (পূর্ণসংখ্যা j = 0; j < n && ok; j++) {
                    ধ্রুবক পূর্ণসংখ্যা l = (পূর্ণসংখ্যা) roundf(GGML_FP16_TO_FP32(prod[j]) / df);
                    ধ্রুবক ggml_half p = GGML_FP32_TO_FP16(df * (দশমিক) l);
                    ok = l >= lmin && l <= lmax && memcmp(&p, &prod[j], আকার(p)) == 0;
                    l_out[j] = l;
                }
                যদি (ok) {
                    *d_out = d;
                    ফেরত সত্য;
                }
            }
        }
    }
    ফেরত মিথ্যা;
}

স্থির শূন্য repack_q3_K_tiled(ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t offset, size_t size) {
    repack_kquant_to_tiled_grid<block_q3_K>(t, তথ্য, offset, size, HTP_TENSOR_TILE_আকার_Q3_K,
        [](uint8_t * tile, ধ্রুবক block_q3_K * b, পূর্ণসংখ্যা row, পূর্ণসংখ্যা kt_local) {
            ধ্রুবক দশমিক d     = GGML_FP16_TO_FP32(b->d);
            uint8_t * lo_pl   = tile;
            uint8_t * neg_pl  = tile + 256;
            ggml_half * sc_pl = (ggml_half *) (tile + 384);

#যদি defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q2_3_k_low2(b->qs, kt_local, lo_pl, row);
            repack_neon_q3_k_hmask(b->hmask, kt_local, neg_pl, row);
#অন্যথা
            জন্য (পূর্ণসংখ্যা lk = 0; lk < 32; lk++) {
                ধ্রুবক পূর্ণসংখ্যা e   = kt_local * 32 + lk;
                ধ্রুবক পূর্ণসংখ্যা g   = lk >> 2;
                ধ্রুবক পূর্ণসংখ্যা pos = row * 4 + (lk & 3);
                lo_pl[(g >> 2) * 128 + pos] |= (uint8_t) (q2_3_K_get_low2(b->qs, e) << ((g & 3) * 2));
                যদি (!q3_K_get_hbit(b, e)) {
                    neg_pl[pos] |= (uint8_t) (1 << g);
                }
            }
#endif
            জন্য (পূর্ণসংখ্যা sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + row] = GGML_FP32_TO_FP16(d * (দশমিক) q3_K_get_scale(b->scales, kt_local * 2 + sub));
            }
        });
}

স্থির শূন্য repack_tiled_q3_K(শূন্য * তথ্য, ধ্রুবক ggml_tensor * t, size_t offset, size_t size) {
    unpack_kquant_from_tiled_grid<block_q3_K>(তথ্য, t, offset, size, HTP_TENSOR_TILE_আকার_Q3_K,
        [](block_q3_K * b, ধ্রুবক uint8_t * sb_tiles, size_t tile_আকার, পূর্ণসংখ্যা row) {
            ggml_half sub_scales[16];
            জন্য (পূর্ণসংখ্যা kt_local = 0; kt_local < 8; kt_local++) {
                ধ্রুবক uint8_t *   tile   = sb_tiles + kt_local * tile_আকার;
                ধ্রুবক uint8_t *   lo_pl  = tile;
                ধ্রুবক uint8_t *   neg_pl = tile + 256;
                ধ্রুবক ggml_half * sc_pl  = (ধ্রুবক ggml_half *) (tile + 384);

                জন্য (পূর্ণসংখ্যা lk = 0; lk < 32; lk++) {
                    ধ্রুবক পূর্ণসংখ্যা e   = kt_local * 32 + lk;
                    ধ্রুবক পূর্ণসংখ্যা g   = lk >> 2;
                    ধ্রুবক পূর্ণসংখ্যা pos = row * 4 + (lk & 3);
                    ধ্রুবক uint8_t lo = (lo_pl[(g >> 2) * 128 + pos] >> ((g & 3) * 2)) & 3;

                    ধ্রুবক পূর্ণসংখ্যা c = e / 128;
                    ধ্রুবক পূর্ণসংখ্যা j = (e % 128) / 32;
                    ধ্রুবক পূর্ণসংখ্যা l = e % 32;
                    b->qs[c * 32 + l] |= (uint8_t) (lo << (2 * j));
                    যদি (!((neg_pl[pos] >> g) & 1)) {
                        b->hmask[l] |= (uint8_t) (1 << (c * 4 + j));
                    }
                }

                জন্য (পূর্ণসংখ্যা sub = 0; sub < 2; sub++) {
                    sub_scales[kt_local * 2 + sub] = sc_pl[sub * 32 + row];
                }
            }

            পূর্ণসংখ্যা ls[16];
            যদি (!hexagon_recover_k_scales(sub_scales, 16, -32, 31, &b->d, ls)) {
                দশমিক max_scale = 0.0f;
                জন্য (পূর্ণসংখ্যা s = 0; s < 16; s++) {
                    যদি (fabsf(GGML_FP16_TO_FP32(sub_scales[s])) > fabsf(max_scale)) {
                        max_scale = GGML_FP16_TO_FP32(sub_scales[s]);
                    }
                }
                b->d = GGML_FP32_TO_FP16(-max_scale / 32.0f);
                ধ্রুবক দশমিক d_actual = GGML_FP16_TO_FP32(b->d);
                ধ্রুবক দশমিক inv_d    = (d_actual != 0.0f) ? (1.0f / d_actual) : 0.0f;
                জন্য (পূর্ণসংখ্যা s = 0; s < 16; s++) {
                    ls[s] = (std::max)(-32, (std::min)(31, (পূর্ণসংখ্যা) roundf(GGML_FP16_TO_FP32(sub_scales[s]) * inv_d)));
                }
            }

            জন্য (পূর্ণসংখ্যা s = 0; s < 16; s++) {
                ধ্রুবক পূর্ণসংখ্যা l = ls[s] + 32;
                যদি (s < 8) {
                    b->scales[s] = l & 0xF;
                } অন্যথা {
                    b->scales[s - 8] |= (uint8_t) ((l & 0xF) << 4);
                }
                b->scales[s % 4 + 8] |= (uint8_t) ((l >> 4) << (2 * (s / 4)));
            }
        });
}

স্থির শূন্য repack_q2_K_tiled(ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t offset, size_t size) {
    repack_kquant_to_tiled_grid<block_q2_K>(t, তথ্য, offset, size, HTP_TENSOR_TILE_আকার_Q2_K,
        [](uint8_t * tile, ধ্রুবক block_q2_K * b, পূর্ণসংখ্যা row, পূর্ণসংখ্যা kt_local) {
            uint8_t * lo_pl   = tile;
            uint8_t * sc_pl   = tile + 256;
            ggml_half * dd_pl = (ggml_half *) (tile + 320);

#যদি defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q2_3_k_low2(b->qs, kt_local, lo_pl, row);
#অন্যথা
            জন্য (পূর্ণসংখ্যা lk = 0; lk < 32; lk++) {
                ধ্রুবক পূর্ণসংখ্যা g   = lk >> 2;
                ধ্রুবক পূর্ণসংখ্যা pos = row * 4 + (lk & 3);
                lo_pl[(g >> 2) * 128 + pos] |= (uint8_t) (q2_3_K_get_low2(b->qs, kt_local * 32 + lk) << ((g & 3) * 2));
            }
#endif
            জন্য (পূর্ণসংখ্যা sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + row] = b->scales[kt_local * 2 + sub];
            }
            যদি (row / 8 == kt_local % 4) {
                dd_pl[row % 8] = kt_local < 4 ? b->d : (ggml_half) (b->dmin ^ 0x8000);
            }
        });
}

স্থির শূন্য repack_tiled_q2_K(শূন্য * তথ্য, ধ্রুবক ggml_tensor * t, size_t offset, size_t size) {
    unpack_kquant_from_tiled_grid<block_q2_K>(তথ্য, t, offset, size, HTP_TENSOR_TILE_আকার_Q2_K,
        [](block_q2_K * b, ধ্রুবক uint8_t * sb_tiles, size_t tile_আকার, পূর্ণসংখ্যা row) {
            জন্য (পূর্ণসংখ্যা kt_local = 0; kt_local < 8; kt_local++) {
                ধ্রুবক uint8_t *   tile  = sb_tiles + kt_local * tile_আকার;
                ধ্রুবক uint8_t *   lo_pl = tile;
                ধ্রুবক uint8_t *   sc_pl = tile + 256;
                ধ্রুবক ggml_half * dd_pl = (ধ্রুবক ggml_half *) (tile + 320);

                জন্য (পূর্ণসংখ্যা lk = 0; lk < 32; lk++) {
                    ধ্রুবক পূর্ণসংখ্যা e   = kt_local * 32 + lk;
                    ধ্রুবক পূর্ণসংখ্যা g   = lk >> 2;
                    ধ্রুবক পূর্ণসংখ্যা pos = row * 4 + (lk & 3);
                    ধ্রুবক uint8_t lo = (lo_pl[(g >> 2) * 128 + pos] >> ((g & 3) * 2)) & 3;
                    b->qs[(e / 128) * 32 + e % 32] |= (uint8_t) (lo << (2 * ((e % 128) / 32)));
                }

                জন্য (পূর্ণসংখ্যা sub = 0; sub < 2; sub++) {
                    b->scales[kt_local * 2 + sub] = sc_pl[sub * 32 + row];
                }
                যদি (row / 8 == kt_local % 4) {
                    যদি (kt_local < 4) {
                        b->d = dd_pl[row % 8];
                    } অন্যথা {
                        b->dmin = (ggml_half) (dd_pl[row % 8] ^ 0x8000);
                    }
                }
            }
        });
}

স্থির অন্তঃরেখ শূন্য get_scale_min_k4(পূর্ণসংখ্যা j, ধ্রুবক uint8_t * q, uint8_t * d, uint8_t * m) {
    যদি (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } অন্যথা {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

স্থির অন্তঃরেখ শূন্য pack_k4_scales_mins(uint8_t * scales, ggml_half * d, ggml_half * dmin,
                                       ধ্রুবক দশমিক * sub_scales, ধ্রুবক দশমিক * sub_mins) {
    দশমিক max_scale = 0.0f;
    দশমিক max_min   = 0.0f;
    জন্য (পূর্ণসংখ্যা j = 0; j < 8; j++) {
        যদি (sub_scales[j] > max_scale) max_scale = sub_scales[j];
        যদি (sub_mins[j]   > max_min)   max_min   = sub_mins[j];
    }

    দশমিক inv_scale = 0.0f;
    যদি (max_scale > 0.0f) {
        *d = GGML_FP32_TO_FP16(max_scale / 63.0f);
        ধ্রুবক দশমিক d_actual = GGML_FP16_TO_FP32(*d);
        inv_scale = (d_actual > 0.0f) ? (1.0f / d_actual) : 0.0f;
    } অন্যথা {
        *d = GGML_FP32_TO_FP16(0.0f);
    }

    দশমিক inv_min = 0.0f;
    যদি (max_min > 0.0f) {
        *dmin = GGML_FP32_TO_FP16(max_min / 63.0f);
        ধ্রুবক দশমিক dmin_actual = GGML_FP16_TO_FP32(*dmin);
        inv_min = (dmin_actual > 0.0f) ? (1.0f / dmin_actual) : 0.0f;
    } অন্যথা {
        *dmin = GGML_FP32_TO_FP16(0.0f);
    }

    জন্য (পূর্ণসংখ্যা j = 0; j < 8; j++) {
        uint8_t ls = (std::min)((uint8_t) 63, (uint8_t) roundf(inv_scale * sub_scales[j]));
        uint8_t lm = (std::min)((uint8_t) 63, (uint8_t) roundf(inv_min * sub_mins[j]));
        যদি (j < 4) {
            scales[j]     = ls;
            scales[j + 4] = lm;
        } অন্যথা {
            scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            scales[j - 4] |= ((ls >> 4) << 6);
            scales[j - 0] |= ((lm >> 4) << 6);
        }
    }
}

template <typename BlockT>
স্থির অন্তঃরেখ শূন্য repack_k4_quants(uint8_t * tile_dst, ধ্রুবক BlockT * b, পূর্ণসংখ্যা row, পূর্ণসংখ্যা kt_local,
                                    int64_t sb_per_row, যৌক্তিক use_neon) {
#যদি defined(__ARM_NEON) || defined(__aarch64__)
    যদি (use_neon) {
        যদি ((row & 7) == 0) {
            ধ্রুবক uint8_t * qs[8];
            জন্য (পূর্ণসংখ্যা rr = 0; rr < 8; ++rr) {
                qs[rr] = b[rr * sb_per_row].qs + (kt_local / 2) * 32;
            }
            repack_neon_k4_rows(qs, tile_dst, row, (kt_local & 1) != 0);
        }
        ফেরত;
    }
#অন্যথা
    GGML_UNUSED(sb_per_row);
    GGML_UNUSED(use_neon);
#endif
    ধ্রুবক uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
    ধ্রুবক পূর্ণসংখ্যা shift = (kt_local & 1) ? 4 : 0;

    জন্য (পূর্ণসংখ্যা cp = 0; cp < 16; cp++) {
        ধ্রুবক uint8_t q0 = (qs_sub[2 * cp + 0] >> shift) & 0x0F;
        ধ্রুবক uint8_t q1 = (qs_sub[2 * cp + 1] >> shift) & 0x0F;
        tile_dst[cp * 32 + row] = (uint8_t) ((q1 << 4) | q0);
    }
}

স্থির শূন্য repack_q4_K_tiled(ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t offset, size_t size) {
    ধ্রুবক int64_t sb_per_row = t->ne[0] / QK_K;
    ধ্রুবক যৌক্তিক    use_neon   = (t->ne[1] % 8 == 0);
    repack_kquant_to_tiled_grid<block_q4_K>(t, তথ্য, offset, size, HTP_TENSOR_TILE_আকার_Q4_K,
        [sb_per_row, use_neon](uint8_t * tile_dst, ধ্রুবক block_q4_K * b, পূর্ণসংখ্যা row, পূর্ণসংখ্যা kt_local) {
            ধ্রুবক দশমিক d    = GGML_FP16_TO_FP32(b->d);
            ধ্রুবক দশমিক dmin = GGML_FP16_TO_FP32(b->dmin);

            uint8_t sc, m;
            get_scale_min_k4(kt_local, b->scales, &sc, &m);

            repack_k4_quants(tile_dst, b, row, kt_local, sb_per_row, use_neon);

            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
            scale_dst[2 * row + 0] = GGML_FP32_TO_FP16(d * (দশমিক) sc);
            scale_dst[2 * row + 1] = GGML_FP32_TO_FP16(-dmin * (দশমিক) m);
        });
}

স্থির শূন্য repack_tiled_q4_K(শূন্য * তথ্য, ধ্রুবক ggml_tensor * t, size_t offset, size_t size) {
    unpack_kquant_from_tiled_grid<block_q4_K>(তথ্য, t, offset, size, HTP_TENSOR_TILE_আকার_Q4_K,
        [](block_q4_K * b, ধ্রুবক uint8_t * sb_tiles, size_t tile_আকার, পূর্ণসংখ্যা row) {
            দশমিক sub_scales[8];
            দশমিক sub_mins[8];

            জন্য (পূর্ণসংখ্যা kt_local = 0; kt_local < 8; kt_local++) {
                ধ্রুবক uint8_t *   tile_src  = sb_tiles + kt_local * tile_আকার;
                ধ্রুবক ggml_half * scale_src = (ধ্রুবক ggml_half *) (tile_src + 512);

                uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
                ধ্রুবক পূর্ণসংখ্যা shift  = (kt_local & 1) ? 4 : 0;

                জন্য (পূর্ণসংখ্যা cp = 0; cp < 16; cp++) {
                    ধ্রুবক uint8_t val = tile_src[cp * 32 + row];
                    qs_sub[2 * cp + 0] |= (uint8_t) ((val & 0x0F) << shift);
                    qs_sub[2 * cp + 1] |= (uint8_t) ((val >> 4)   << shift);
                }

                ধ্রুবক দশমিক D = GGML_FP16_TO_FP32(scale_src[2 * row + 0]);
                ধ্রুবক দশমিক M = GGML_FP16_TO_FP32(scale_src[2 * row + 1]);
                sub_scales[kt_local] = (D > 0.0f) ? D : 0.0f;
                sub_mins[kt_local]   = (-M > 0.0f) ? -M : 0.0f;
            }

            pack_k4_scales_mins(b->scales, &b->d, &b->dmin, sub_scales, sub_mins);
        });
}

স্থির শূন্য repack_q5_K_tiled(ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t offset, size_t size) {
    ধ্রুবক int64_t sb_per_row = t->ne[0] / QK_K;
    ধ্রুবক যৌক্তিক    use_neon   = (t->ne[1] % 8 == 0);
    repack_kquant_to_tiled_grid<block_q5_K>(t, তথ্য, offset, size, HTP_TENSOR_TILE_আকার_Q5_K,
        [sb_per_row, use_neon](uint8_t * tile_dst, ধ্রুবক block_q5_K * b, পূর্ণসংখ্যা row, পূর্ণসংখ্যা kt_local) {
            ধ্রুবক দশমিক d     = GGML_FP16_TO_FP32(b->d);
            ধ্রুবক দশমিক dmin  = GGML_FP16_TO_FP32(b->dmin);
            uint8_t * plane   = tile_dst + 640;

            uint8_t sc, m;
            get_scale_min_k4(kt_local, b->scales, &sc, &m);

            repack_k4_quants(tile_dst, b, row, kt_local, sb_per_row, use_neon);

            ধ্রুবক uint8_t hbit = (uint8_t) (1 << kt_local);
#যদি defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q5_k_qh(b->qh, hbit, plane, row);
#অন্যথা
            জন্য (পূর্ণসংখ্যা cp = 0; cp < 16; cp++) {
                ধ্রুবক পূর্ণসংখ্যা i    = cp / 4;
                ধ্রুবক পূর্ণসংখ্যা lane = (cp % 4) * 32 + row;
                যদি (b->qh[2 * cp + 0] & hbit) {
                    plane[lane] |= (uint8_t) (1 << (2 * i));
                }
                যদি (b->qh[2 * cp + 1] & hbit) {
                    plane[lane] |= (uint8_t) (1 << (2 * i + 1));
                }
            }
#endif

            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
            scale_dst[2 * row + 0] = GGML_FP32_TO_FP16(d * (দশমিক) sc);
            scale_dst[2 * row + 1] = GGML_FP32_TO_FP16(-dmin * (দশমিক) m);
        });
}

স্থির শূন্য repack_tiled_q5_K(শূন্য * তথ্য, ধ্রুবক ggml_tensor * t, size_t offset, size_t size) {
    unpack_kquant_from_tiled_grid<block_q5_K>(তথ্য, t, offset, size, HTP_TENSOR_TILE_আকার_Q5_K,
        [](block_q5_K * b, ধ্রুবক uint8_t * sb_tiles, size_t tile_আকার, পূর্ণসংখ্যা row) {
            দশমিক sub_scales[8];
            দশমিক sub_mins[8];

            জন্য (পূর্ণসংখ্যা kt_local = 0; kt_local < 8; kt_local++) {
                ধ্রুবক uint8_t *   tile_src  = sb_tiles + kt_local * tile_আকার;
                ধ্রুবক uint8_t *   plane     = tile_src + 640;
                ধ্রুবক ggml_half * scale_src = (ধ্রুবক ggml_half *) (tile_src + 512);

                uint8_t * qs_sub   = b->qs + (kt_local / 2) * 32;
                ধ্রুবক পূর্ণসংখ্যা shift    = (kt_local & 1) ? 4 : 0;
                ধ্রুবক uint8_t hbit = (uint8_t) (1 << kt_local);

                জন্য (পূর্ণসংখ্যা cp = 0; cp < 16; cp++) {
                    ধ্রুবক uint8_t val = tile_src[cp * 32 + row];
                    qs_sub[2 * cp + 0] |= (uint8_t) ((val & 0x0F) << shift);
                    qs_sub[2 * cp + 1] |= (uint8_t) ((val >> 4)   << shift);

                    ধ্রুবক পূর্ণসংখ্যা i    = cp / 4;
                    ধ্রুবক পূর্ণসংখ্যা lane = (cp % 4) * 32 + row;
                    যদি (plane[lane] & (1 << (2 * i))) {
                        b->qh[2 * cp + 0] |= hbit;
                    }
                    যদি (plane[lane] & (1 << (2 * i + 1))) {
                        b->qh[2 * cp + 1] |= hbit;
                    }
                }

                ধ্রুবক দশমিক D = GGML_FP16_TO_FP32(scale_src[2 * row + 0]);
                ধ্রুবক দশমিক M = GGML_FP16_TO_FP32(scale_src[2 * row + 1]);
                sub_scales[kt_local] = (D > 0.0f) ? D : 0.0f;
                sub_mins[kt_local]   = (-M > 0.0f) ? -M : 0.0f;
            }

            pack_k4_scales_mins(b->scales, &b->d, &b->dmin, sub_scales, sub_mins);
        });
}

স্থির অন্তঃরেখ শূন্য q1_0_tiled_bit_location(পূর্ণসংখ্যা k, পূর্ণসংখ্যা row, পূর্ণসংখ্যা & byte_offset, পূর্ণসংখ্যা & bit) {
    ধ্রুবক পূর্ণসংখ্যা quarter = k / 32;
    ধ্রুবক পূর্ণসংখ্যা kq      = k % 32;

    byte_offset = quarter * 128 + 4 * row + (kq % 4);
    bit         = kq / 4;
}

স্থির শূন্য repack_q1_0_tiled(ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t offset, size_t size) {
    ধ্রুবক block_q1_0 * src_matrix = (ধ্রুবক block_q1_0 *) তথ্য;
    ধ্রুবক int64_t ne0 = t->ne[0];
    ধ্রুবক int64_t ne1 = t->ne[1];
    ধ্রুবক int64_t ne2 = t->ne[2];
    ধ্রুবক int64_t ne3 = t->ne[3];
    ধ্রুবক পূর্ণসংখ্যা n_col_tiles = hex_round_up(ne1, 32) / 32;
    ধ্রুবক পূর্ণসংখ্যা n_k_tiles = ne0 / QK1_0;
    ধ্রুবক size_t tile_আকার = HTP_TENSOR_TILE_আকার_Q1_0;
    ধ্রুবক size_t matrix_আকার = (size_t) n_col_tiles * n_k_tiles * tile_আকার;
    ধ্রুবক size_t slice_আকার = ne1 * ggml_row_আকার(t->type, ne0);
    ধ্রুবক int64_t start_slice = offset / slice_আকার;
    ধ্রুবক int64_t end_slice = (std::min)((int64_t) ((offset + size + slice_আকার - 1) / slice_আকার), ne2 * ne3);

    GGML_ASSERT(ne0 % QK1_0 == 0);

    জন্য (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        ধ্রুবক block_q1_0 * src_slice = src_matrix + (slice_idx - start_slice) * ne1 * n_k_tiles;
        uint8_t * matrix_dst = (uint8_t *) t->তথ্য + slice_idx * matrix_আকার;

        repack_parallel_for_ct(matrix_আকার, n_col_tiles, [&](পূর্ণসংখ্যা ct0, পূর্ণসংখ্যা ct1) {
            জন্য (পূর্ণসংখ্যা ct = ct0; ct < ct1; ct++) {
                জন্য (পূর্ণসংখ্যা kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = matrix_dst + ((size_t) ct * n_k_tiles + kt) * tile_আকার;
                    ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
                    memset(tile_dst, 0, tile_আকার);

                    জন্য (পূর্ণসংখ্যা row = 0; row < 32; row++) {
                        ধ্রুবক int64_t r = ct * 32 + row;
                        যদি (r >= ne1) {
                            চালু;
                        }

                        ধ্রুবক block_q1_0 & block = src_slice[r * n_k_tiles + kt];
                        scale_dst[row] = block.d;
#যদি defined(__ARM_NEON) || defined(__aarch64__)
                        repack_neon_q1_0_block(block.qs, tile_dst, row);
#অন্যথা
                        জন্য (পূর্ণসংখ্যা k = 0; k < QK1_0; k++) {
                            পূর্ণসংখ্যা byte_offset;
                            পূর্ণসংখ্যা bit;
                            q1_0_tiled_bit_location(k, row, byte_offset, bit);
                            যদি (block.qs[k / 8] & (1u << (k % 8))) {
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

স্থির শূন্য repack_tiled_q1_0(শূন্য * তথ্য, ধ্রুবক ggml_tensor * t, size_t offset, size_t size) {
    block_q1_0 * dst_matrix = (block_q1_0 *) তথ্য;
    ধ্রুবক int64_t ne0 = t->ne[0];
    ধ্রুবক int64_t ne1 = t->ne[1];
    ধ্রুবক int64_t ne2 = t->ne[2];
    ধ্রুবক int64_t ne3 = t->ne[3];
    ধ্রুবক পূর্ণসংখ্যা n_col_tiles = hex_round_up(ne1, 32) / 32;
    ধ্রুবক পূর্ণসংখ্যা n_k_tiles = ne0 / QK1_0;
    ধ্রুবক size_t tile_আকার = HTP_TENSOR_TILE_আকার_Q1_0;
    ধ্রুবক size_t matrix_আকার = (size_t) n_col_tiles * n_k_tiles * tile_আকার;
    ধ্রুবক size_t row_আকার_bytes = ggml_row_আকার(t->type, ne0);
    ধ্রুবক size_t slice_আকার = ne1 * row_আকার_bytes;
    ধ্রুবক int64_t start_slice = offset / slice_আকার;
    ধ্রুবক int64_t end_slice = (std::min)((int64_t) ((offset + size + slice_আকার - 1) / slice_আকার), ne2 * ne3);

    GGML_ASSERT(ne0 % QK1_0 == 0);

    জন্য (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        ধ্রুবক size_t cur_start_byte = (std::max)(offset, (size_t) slice_idx * slice_আকার);
        ধ্রুবক size_t cur_end_byte = (std::min)(offset + size, (size_t) (slice_idx + 1) * slice_আকার);
        ধ্রুবক int64_t start_row = (cur_start_byte - (size_t) slice_idx * slice_আকার) / row_আকার_bytes;
        ধ্রুবক int64_t end_row = (std::min)((int64_t) ((cur_end_byte - (size_t) slice_idx * slice_আকার + row_আকার_bytes - 1) / row_আকার_bytes), ne1);
        block_q1_0 * dst_slice = dst_matrix + (cur_start_byte - offset) / আকার(block_q1_0);
        ধ্রুবক uint8_t * matrix_src = (ধ্রুবক uint8_t *) t->তথ্য + slice_idx * matrix_আকার;

        জন্য (পূর্ণসংখ্যা ct = start_row / 32; ct < (end_row + 31) / 32; ct++) {
            জন্য (পূর্ণসংখ্যা kt = 0; kt < n_k_tiles; kt++) {
                ধ্রুবক uint8_t * tile_src = matrix_src + ((size_t) ct * n_k_tiles + kt) * tile_আকার;
                ধ্রুবক ggml_half * scale_src = (ধ্রুবক ggml_half *) (tile_src + 512);

                জন্য (পূর্ণসংখ্যা row = 0; row < 32; row++) {
                    ধ্রুবক int64_t r = ct * 32 + row;
                    যদি (r < start_row || r >= end_row) {
                        চালু;
                    }

                    block_q1_0 & block = dst_slice[(r - start_row) * n_k_tiles + kt];
                    memset(block.qs, 0, আকার(block.qs));
                    block.d = scale_src[row];
                    জন্য (পূর্ণসংখ্যা k = 0; k < QK1_0; k++) {
                        পূর্ণসংখ্যা byte_offset;
                        পূর্ণসংখ্যা bit;
                        q1_0_tiled_bit_location(k, row, byte_offset, bit);
                        যদি (tile_src[byte_offset] & (1u << bit)) {
                            block.qs[k / 8] |= (uint8_t) (1u << (k % 8));
                        }
                    }
                }
            }
        }
    }
}

// --- Top-level Repack / Unpack Dispatchers ---

শূন্য htp_repack_tensor_tiled(ggml_tensor * tensor, ধ্রুবক শূন্য * তথ্য, size_t size) {
    নির্বাচন (tensor->type) {
        ক্ষেত্র GGML_TYPE_Q1_0:
            repack_q1_0_tiled(tensor, তথ্য, 0, size);
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q4_0:
        ক্ষেত্র GGML_TYPE_IQ4_NL:
            repack_q4_0_tiled(tensor, তথ্য, 0, size);
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q4_1:
            repack_q4_1_tiled(tensor, তথ্য, 0, size);
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q8_0:
            repack_q8_0_tiled(tensor, তথ্য, 0, size);
            বিরতি;
        ক্ষেত্র GGML_TYPE_MXFP4:
            repack_mxfp4_tiled(tensor, তথ্য, 0, size);
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q2_0:
            repack_q2_0_tiled(tensor, তথ্য, 0, size);
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q2_K:
            repack_q2_K_tiled(tensor, তথ্য, 0, size);
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q3_K:
            repack_q3_K_tiled(tensor, তথ্য, 0, size);
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q4_K:
            repack_q4_K_tiled(tensor, তথ্য, 0, size);
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q5_K:
            repack_q5_K_tiled(tensor, তথ্য, 0, size);
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q6_K:
            repack_q6_K_tiled(tensor, তথ্য, 0, size);
            বিরতি;
        ক্ষেত্র GGML_TYPE_F16:
            repack_float_to_f16_tiled<uint16_t>(tensor, তথ্য, 0, size,
                [](uint16_t v) { ফেরত v; });
            বিরতি;
        ক্ষেত্র GGML_TYPE_F32:
            repack_float_to_f16_tiled<দশমিক>(tensor, তথ্য, 0, size,
                [](দশমিক v) { ফেরত (uint16_t) GGML_FP32_TO_FP16(v); });
            বিরতি;
        ক্ষেত্র GGML_TYPE_BF16:
            repack_float_to_f16_tiled<ggml_bf16_t>(tensor, তথ্য, 0, size,
                [](ggml_bf16_t v) { ফেরত (uint16_t) GGML_FP32_TO_FP16(GGML_BF16_TO_FP32(v)); });
            বিরতি;
        পূর্বনির্ধারিত:
            বিরতি;
    }
}

শূন্য htp_unpack_tensor_tiled(শূন্য * তথ্য, ধ্রুবক ggml_tensor * tensor, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor));

    নির্বাচন (tensor->type) {
        ক্ষেত্র GGML_TYPE_Q1_0:
            repack_tiled_q1_0(তথ্য, tensor, offset, size);
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q4_0:
        ক্ষেত্র GGML_TYPE_IQ4_NL:
            repack_tiled_q4_0(তথ্য, tensor, offset, size);
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q4_1:
            repack_tiled_q4_1(তথ্য, tensor, offset, size);
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q8_0:
            repack_tiled_q8_0(তথ্য, tensor, offset, size);
            বিরতি;
        ক্ষেত্র GGML_TYPE_MXFP4:
            repack_tiled_mxfp4(তথ্য, tensor, offset, size);
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q2_0:
            repack_tiled_q2_0(তথ্য, tensor, offset, size);
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q2_K:
            repack_tiled_q2_K(তথ্য, tensor, offset, size);
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q3_K:
            repack_tiled_q3_K(তথ্য, tensor, offset, size);
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q4_K:
            repack_tiled_q4_K(তথ্য, tensor, offset, size);
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q5_K:
            repack_tiled_q5_K(তথ্য, tensor, offset, size);
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q6_K:
            repack_tiled_q6_K(তথ্য, tensor, offset, size);
            বিরতি;
        ক্ষেত্র GGML_TYPE_F16:
            unpack_f16_tiled_to_float<uint16_t>(তথ্য, tensor, offset, size,
                [](uint16_t v) { ফেরত v; });
            বিরতি;
        ক্ষেত্র GGML_TYPE_F32:
            unpack_f16_tiled_to_float<দশমিক>(তথ্য, tensor, offset, size,
                [](uint16_t v) { ফেরত ggml_fp16_to_fp32((ggml_fp16_t) v); });
            বিরতি;
        ক্ষেত্র GGML_TYPE_BF16:
            unpack_f16_tiled_to_float<ggml_bf16_t>(তথ্য, tensor, offset, size,
                [](uint16_t v) { ফেরত ggml_fp32_to_bf16(ggml_fp16_to_fp32((ggml_fp16_t) v)); });
            বিরতি;
        পূর্বনির্ধারিত:
            memcpy(তথ্য, (ধ্রুবক অক্ষর *) tensor->তথ্য + offset, size);
            বিরতি;
    }
}
