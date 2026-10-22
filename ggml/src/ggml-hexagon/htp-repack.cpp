#شامل "htp-repack.h"

#تعریف_کرو GGML_COMMON_IMPL_CPP
#شامل "ggml-common.h"
#شامل "ggml-impl.h"
#شامل "ggml-quants.h"

#شامل <algorithm>
#شامل <cmath>
#شامل <cstring>
#شامل <initializer_list>
#شامل <thread>

#اگر defined(__ARM_NEON) || defined(__aarch64__)
#    include <arm_neon.h>
#اگر_ختم

// --- 4-bit block-32 nibble helpers (shared by Q4_0, IQ4_NL, Q4_1, MXFP4) ---

ساکن اندرونی خالی unpack_nibbles_32(uint8_t * dst, ثابت uint8_t * qs) {
    برائے (غیرعلامتی صحیح i = 0; i < 16; ++i) {
        dst[i + 0]  = qs[i] & 0x0F;
        dst[i + 16] = qs[i] >> 4;
    }
}

ساکن اندرونی خالی pack_nibbles_32(uint8_t * qs, ثابت uint8_t * src) {
    برائے (غیرعلامتی صحیح i = 0; i < 16; ++i) {
        qs[i] = src[i + 0] | (src[i + 16] << 4);
    }
}

ساکن اندرونی خالی transpose_nibbles_مقصد_tile(uint8_t * tile_dst, ثابت uint8_t tile_quants[32][32]) {
    برائے (صحیح cp = 0; cp < 16; cp++) {
        برائے (صحیح صف = 0; صف < 32; صف++) {
            tile_dst[cp * 32 + صف] = (tile_quants[صف][2 * cp + 1] << 4) | tile_quants[صف][2 * cp];
        }
    }
}

ساکن اندرونی خالی transpose_tile_مقصد_nibbles(uint8_t tile_quants[32][32], ثابت uint8_t * tile_src) {
    برائے (صحیح cp = 0; cp < 16; cp++) {
        برائے (صحیح صف = 0; صف < 32; صف++) {
            ثابت uint8_t val = tile_src[cp * 32 + صف];
            tile_quants[صف][2 * cp + 0] = val & 0x0F;
            tile_quants[صف][2 * cp + 1] = val >> 4;
        }
    }
}

#اگر defined(__ARM_NEON) || defined(__aarch64__)
ساکن خالی repack_neon_transpose8x8(ثابت uint8x8_t صفیں[8], uint8x8_t columns[8]) {
    ثابت uint8x8x2_t t0 = vtrn_u8(صفیں[0], صفیں[1]);
    ثابت uint8x8x2_t t1 = vtrn_u8(صفیں[2], صفیں[3]);
    ثابت uint8x8x2_t t2 = vtrn_u8(صفیں[4], صفیں[5]);
    ثابت uint8x8x2_t t3 = vtrn_u8(صفیں[6], صفیں[7]);
    ثابت uint16x4x2_t t4 = vtrn_u16(vreinterpret_u16_u8(t0.val[0]), vreinterpret_u16_u8(t1.val[0]));
    ثابت uint16x4x2_t t5 = vtrn_u16(vreinterpret_u16_u8(t0.val[1]), vreinterpret_u16_u8(t1.val[1]));
    ثابت uint16x4x2_t t6 = vtrn_u16(vreinterpret_u16_u8(t2.val[0]), vreinterpret_u16_u8(t3.val[0]));
    ثابت uint16x4x2_t t7 = vtrn_u16(vreinterpret_u16_u8(t2.val[1]), vreinterpret_u16_u8(t3.val[1]));
    ثابت uint32x2x2_t t8 = vtrn_u32(vreinterpret_u32_u16(t4.val[0]), vreinterpret_u32_u16(t6.val[0]));
    ثابت uint32x2x2_t t9 = vtrn_u32(vreinterpret_u32_u16(t4.val[1]), vreinterpret_u32_u16(t6.val[1]));
    ثابت uint32x2x2_t ta = vtrn_u32(vreinterpret_u32_u16(t5.val[0]), vreinterpret_u32_u16(t7.val[0]));
    ثابت uint32x2x2_t tb = vtrn_u32(vreinterpret_u32_u16(t5.val[1]), vreinterpret_u32_u16(t7.val[1]));
    columns[0] = vreinterpret_u8_u32(t8.val[0]);
    columns[1] = vreinterpret_u8_u32(ta.val[0]);
    columns[2] = vreinterpret_u8_u32(t9.val[0]);
    columns[3] = vreinterpret_u8_u32(tb.val[0]);
    columns[4] = vreinterpret_u8_u32(t8.val[1]);
    columns[5] = vreinterpret_u8_u32(ta.val[1]);
    columns[6] = vreinterpret_u8_u32(t9.val[1]);
    columns[7] = vreinterpret_u8_u32(tb.val[1]);
}

ساکن uint8x8_t repack_neon_pack_4bit_جوڑے(uint8x8_t پہلا, uint8x8_t second, منطقی بلند) {
    ثابت uint8x8x2_t separated = vuzp_u8(پہلا, second);
    ثابت uint8x8_t نقاب = vdup_n_u8(0x0F);
    uint8x8_t even = separated.val[0];
    uint8x8_t odd  = separated.val[1];

    اگر (بلند) {
        even = vshr_n_u8(even, 4);
        odd  = vshr_n_u8(odd, 4);
    }

    even = vand_u8(even, نقاب);
    odd  = vand_u8(odd, نقاب);
    واپس vorr_u8(vshl_n_u8(odd, 4), even);
}

ساکن خالی repack_neon_4bit_صفیں(ثابت uint8_t * qs[8], uint8_t * tile_dst, صحیح row0) {
    uint8x8_t کم_صفیں[8];
    uint8x8_t بلند_صفیں[8];
    uint8x8_t columns[8];

    برائے (صحیح rr = 0; rr < 8; ++rr) {
        ثابت uint8x16_t packed = vld1q_u8(qs[rr]);
        ثابت uint8x8_t پہلا = vget_کم_u8(packed);
        ثابت uint8x8_t second = vget_بلند_u8(packed);
        کم_صفیں[rr]  = repack_neon_pack_4bit_جوڑے(پہلا, second, جھوٹ);
        بلند_صفیں[rr] = repack_neon_pack_4bit_جوڑے(پہلا, second, سچ);
    }

    repack_neon_transpose8x8(کم_صفیں, columns);
    برائے (صحیح cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + cp * 32 + row0, columns[cp]);
    }

    repack_neon_transpose8x8(بلند_صفیں, columns);
    برائے (صحیح cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + (cp + 8) * 32 + row0, columns[cp]);
    }
}

ساکن خالی repack_neon_k4_صفیں(ثابت uint8_t * qs[8], uint8_t * tile_dst, صحیح row0, منطقی بلند) {
    uint8x8_t پہلا_صفیں[8];
    uint8x8_t دوسرا_صفیں[8];
    uint8x8_t columns[8];

    برائے (صحیح rr = 0; rr < 8; ++rr) {
        پہلا_صفیں[rr] = repack_neon_pack_4bit_جوڑے(vld1_u8(qs[rr]), vld1_u8(qs[rr] + 8), بلند);
        دوسرا_صفیں[rr] = repack_neon_pack_4bit_جوڑے(vld1_u8(qs[rr] + 16), vld1_u8(qs[rr] + 24), بلند);
    }

    repack_neon_transpose8x8(پہلا_صفیں, columns);
    برائے (صحیح cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + cp * 32 + row0, columns[cp]);
    }

    repack_neon_transpose8x8(دوسرا_صفیں, columns);
    برائے (صحیح cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + (cp + 8) * 32 + row0, columns[cp]);
    }
}

ساکن خالی repack_neon_q8_صفیں(ثابت uint8_t * qs[8], uint8_t * tile_dst, صحیح row0) {
    uint8x8_t صفیں[8];
    uint8x8_t columns[8];

    برائے (صحیح cp = 0; cp < 16; cp += 4) {
        برائے (صحیح rr = 0; rr < 8; ++rr) {
            صفیں[rr] = vld1_u8(qs[rr] + 2 * cp);
        }
        repack_neon_transpose8x8(صفیں, columns);

        برائے (صحیح pair = 0; pair < 4; ++pair) {
            ثابت uint8x8x2_t interleaved = vzip_u8(columns[2 * pair], columns[2 * pair + 1]);
            uint8_t * dst = tile_dst + (cp + pair) * 64 + 2 * row0;
            vst1_u8(dst, interleaved.val[0]);
            vst1_u8(dst + 8, interleaved.val[1]);
        }
    }
}

ساکن اندرونی خالی repack_store_u32(uint8_t * dst, uint32_t v) {
    حافظہ_نقل(dst, &v, حجم(v));
}

ساکن اندرونی uint32x4_t repack_neon_transpose_2bit_4x4(uint32x4_t w) {
    uint32x4_t t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 6)), vdupq_n_u32(0x00CC00CCu));
    w = veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 6)));
    t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 12)), vdupq_n_u32(0x0000F0F0u));
    واپس veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 12)));
}

ساکن اندرونی خالی repack_neon_q2_0_بلاک(ثابت uint8_t * qs, uint8_t * tile, صحیح صف) {
    ثابت uint32x4_t w = repack_neon_transpose_2bit_4x4(vreinterpretq_u32_u8(vld1q_u8(qs)));
    repack_store_u32(tile + 0 * 128 + 4 * صف, vgetq_lane_u32(w, 0));
    repack_store_u32(tile + 1 * 128 + 4 * صف, vgetq_lane_u32(w, 1));
    repack_store_u32(tile + 2 * 128 + 4 * صف, vgetq_lane_u32(w, 2));
    repack_store_u32(tile + 3 * 128 + 4 * صف, vgetq_lane_u32(w, 3));
}

ساکن اندرونی خالی repack_neon_q1_0_بلاک(ثابت uint8_t * qs, uint8_t * tile, صحیح صف) {
    uint32x4_t w = vreinterpretq_u32_u8(vld1q_u8(qs));
    uint32x4_t t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 1)), vdupq_n_u32(0x22222222u));
    w = veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 1)));
    t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 3)), vdupq_n_u32(0x0A0A0A0Au));
    w = veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 3)));
    w = repack_neon_transpose_2bit_4x4(w);
    repack_store_u32(tile + 0 * 128 + 4 * صف, vgetq_lane_u32(w, 0));
    repack_store_u32(tile + 1 * 128 + 4 * صف, vgetq_lane_u32(w, 1));
    repack_store_u32(tile + 2 * 128 + 4 * صف, vgetq_lane_u32(w, 2));
    repack_store_u32(tile + 3 * 128 + 4 * صف, vgetq_lane_u32(w, 3));
}

ساکن اندرونی خالی repack_neon_pack_2bit_plane(uint8x16_t v0, uint8x16_t v1, uint8_t * dst, صحیح صف) {
    ثابت uint32x4x2_t z   = vuzpq_u32(vreinterpretq_u32_u8(v0), vreinterpretq_u32_u8(v1));
    ثابت uint32x4_t   p01 = vorrq_u32(z.val[0], vshlq_n_u32(z.val[1], 2));
    ثابت uint32x2x2_t u   = vuzp_u32(vget_کم_u32(p01), vget_بلند_u32(p01));
    ثابت uint32x2_t   برآمد = vorr_u32(u.val[0], vshl_n_u32(u.val[1], 4));
    repack_store_u32(dst + 0 * 128 + صف * 4, vget_lane_u32(برآمد, 0));
    repack_store_u32(dst + 1 * 128 + صف * 4, vget_lane_u32(برآمد, 1));
}

ساکن اندرونی خالی repack_neon_q2_3_k_low2(ثابت uint8_t * qs, صحیح kt_local, uint8_t * lo_pl, صحیح صف) {
    ثابت uint8_t *  qs_sub = qs + (kt_local / 4) * 32;
    ثابت int8x16_t  سرکاؤ  = vdupq_n_s8(-2 * (kt_local % 4));
    ثابت uint8x16_t نقاب   = vdupq_n_u8(0x03);
    ثابت uint8x16_t v0     = vandq_u8(vshlq_u8(vld1q_u8(qs_sub +  0), سرکاؤ), نقاب);
    ثابت uint8x16_t v1     = vandq_u8(vshlq_u8(vld1q_u8(qs_sub + 16), سرکاؤ), نقاب);
    repack_neon_pack_2bit_plane(v0, v1, lo_pl, صف);
}

ساکن اندرونی خالی repack_neon_q3_k_hmask(ثابت uint8_t * hmask, صحیح kt_local, uint8_t * neg_pl, صحیح صف) {
    ساکن ثابت uint32_t نقاب_الفاظ[4] = { 0x01010101u, 0x02020202u, 0x04040404u, 0x08080808u };
    ثابت uint8x16_t m0   = vreinterpretq_u8_u32(vld1q_u32(نقاب_الفاظ));
    ثابت uint8x16_t m1   = vshlq_n_u8(m0, 4);
    ثابت uint8x16_t hbit = vdupq_n_u8((uint8_t) (1u << kt_local));
    ثابت uint8x16_t b0   = vbicq_u8(m0, vtstq_u8(vld1q_u8(hmask +  0), hbit));
    ثابت uint8x16_t b1   = vbicq_u8(m1, vtstq_u8(vld1q_u8(hmask + 16), hbit));
    ثابت uint32x4_t b01  = vreinterpretq_u32_u8(vorrq_u8(b0, b1));
    ثابت uint32x2_t f0   = vorr_u32(vget_کم_u32(b01), vget_بلند_u32(b01));
    ثابت uint32x2_t f1   = vpadd_u32(f0, f0);
    repack_store_u32(neg_pl + صف * 4, vget_lane_u32(f1, 0));
}

ساکن اندرونی خالی repack_neon_q6_k(ثابت بلاک_q6_K * b, صحیح kt_local, uint8_t * lo_pl, uint8_t * hi_pl, صحیح صف) {
    ثابت صحیح        c      = kt_local / 4;
    ثابت صحیح        g_sb   = kt_local % 4;
    ثابت uint8_t *  ql     = b->ql + c * 64 + (g_sb & 1) * 32;
    ثابت uint8_t *  qh     = b->qh + c * 32;
    ثابت int8x16_t  lshift = vdupq_n_s8((g_sb >= 2) ? -4 : 0);
    ثابت int8x16_t  hshift = vdupq_n_s8(-2 * g_sb);
    ثابت uint8x16_t mask4  = vdupq_n_u8(0x0F);
    ثابت uint8x16_t mask2  = vdupq_n_u8(0x03);

    ثابت uint8x16_t lo0  = vandq_u8(vshlq_u8(vld1q_u8(ql +  0), lshift), mask4);
    ثابت uint8x16_t lo1  = vandq_u8(vshlq_u8(vld1q_u8(ql + 16), lshift), mask4);
    ثابت uint32x4x2_t lz = vuzpq_u32(vreinterpretq_u32_u8(lo0), vreinterpretq_u32_u8(lo1));
    ثابت uint32x4_t lout = vorrq_u32(lz.val[0], vshlq_n_u32(lz.val[1], 4));
    repack_store_u32(lo_pl + 0 * 128 + صف * 4, vgetq_lane_u32(lout, 0));
    repack_store_u32(lo_pl + 1 * 128 + صف * 4, vgetq_lane_u32(lout, 1));
    repack_store_u32(lo_pl + 2 * 128 + صف * 4, vgetq_lane_u32(lout, 2));
    repack_store_u32(lo_pl + 3 * 128 + صف * 4, vgetq_lane_u32(lout, 3));

    ثابت uint8x16_t hi0 = vandq_u8(vshlq_u8(vld1q_u8(qh +  0), hshift), mask2);
    ثابت uint8x16_t hi1 = vandq_u8(vshlq_u8(vld1q_u8(qh + 16), hshift), mask2);
    repack_neon_pack_2bit_plane(hi0, hi1, hi_pl, صف);
}

ساکن اندرونی خالی repack_neon_q5_k_qh(ثابت uint8_t * qh, uint8_t hbit, uint8_t * plane, صحیح صف) {
    ساکن ثابت uint8_t نقاب_بائٹیں[16] = { 1, 2, 1, 2, 1, 2, 1, 2, 4, 8, 4, 8, 4, 8, 4, 8 };
    ثابت uint8x16_t m0   = vld1q_u8(نقاب_بائٹیں);
    ثابت uint8x16_t m1   = vshlq_n_u8(m0, 4);
    ثابت uint8x16_t vbit = vdupq_n_u8(hbit);
    ثابت uint8x16_t b0   = vandq_u8(vtstq_u8(vld1q_u8(qh +  0), vbit), m0);
    ثابت uint8x16_t b1   = vandq_u8(vtstq_u8(vld1q_u8(qh + 16), vbit), m1);
    ثابت uint8x16_t b01  = vorrq_u8(b0, b1);
    ثابت uint8x8_t  fold = vorr_u8(vget_کم_u8(b01), vget_بلند_u8(b01));
    ثابت uint8x8_t  برآمد  = vpadd_u8(fold, fold);
    plane[0 * 32 + صف] = vget_lane_u8(برآمد, 0);
    plane[1 * 32 + صف] = vget_lane_u8(برآمد, 1);
    plane[2 * 32 + صف] = vget_lane_u8(برآمد, 2);
    plane[3 * 32 + صف] = vget_lane_u8(برآمد, 3);
}

ساکن اندرونی uint16x8_t repack_neon_بھرو_f16x8(ثابت uint16_t * src) {
    واپس vld1q_u16(src);
}

#اگر defined(__aarch64__) || defined(__ARM_FP16_FORMAT_IEEE)
ساکن اندرونی uint16x8_t repack_neon_بھرو_f16x8(ثابت اعشاری * src) {
    ثابت float16x4_t lo = vcvt_f16_f32(vld1q_f32(src));
    ثابت float16x4_t hi = vcvt_f16_f32(vld1q_f32(src + 4));
    واپس vreinterpretq_u16_f16(vcombine_f16(lo, hi));
}

ساکن اندرونی uint16x8_t repack_neon_بھرو_f16x8(ثابت ggml_bf16_t * src) {
    ثابت uint16x8_t  bf   = vld1q_u16((ثابت uint16_t *) src);
    ثابت float32x4_t f_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_کم_u16(bf), 16));
    ثابت float32x4_t f_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_بلند_u16(bf), 16));
    ثابت float16x4_t lo   = vcvt_f16_f32(f_lo);
    ثابت float16x4_t hi   = vcvt_f16_f32(f_hi);
    واپس vreinterpretq_u16_f16(vcombine_f16(lo, hi));
}
#ورنہ
ساکن اندرونی uint16x8_t repack_neon_بھرو_f16x8(ثابت اعشاری * src) {
    uint16_t tmp[8];
    برائے (صحیح i = 0; i < 8; ++i) {
        tmp[i] = GGML_FP32_مقصد_FP16(src[i]);
    }
    واپس vld1q_u16(tmp);
}

ساکن اندرونی uint16x8_t repack_neon_بھرو_f16x8(ثابت ggml_bf16_t * src) {
    uint16_t tmp[8];
    برائے (صحیح i = 0; i < 8; ++i) {
        tmp[i] = GGML_FP32_مقصد_FP16(GGML_BF16_مقصد_FP32(src[i]));
    }
    واپس vld1q_u16(tmp);
}
#اگر_ختم

سانچہ <قسم_نام SrcT>
ساکن خالی repack_neon_f16_tile(uint16_t * tile_dst, ثابت SrcT * src_قاش, int64_t صف_stride, صحیح ct, صحیح kt) {
    برائے (صحیح row0 = 0; row0 < 32; row0 += 4) {
        ثابت SrcT * r0 = src_قاش + (int64_t) (ct * 32 + row0 + 0) * صف_stride + kt * 32;
        ثابت SrcT * r1 = src_قاش + (int64_t) (ct * 32 + row0 + 1) * صف_stride + kt * 32;
        ثابت SrcT * r2 = src_قاش + (int64_t) (ct * 32 + row0 + 2) * صف_stride + kt * 32;
        ثابت SrcT * r3 = src_قاش + (int64_t) (ct * 32 + row0 + 3) * صف_stride + kt * 32;
        برائے (صحیح cp = 0; cp < 16; cp += 4) {
            ثابت uint32x4_t v0 = vreinterpretq_u32_u16(repack_neon_بھرو_f16x8(r0 + 2 * cp));
            ثابت uint32x4_t v1 = vreinterpretq_u32_u16(repack_neon_بھرو_f16x8(r1 + 2 * cp));
            ثابت uint32x4_t v2 = vreinterpretq_u32_u16(repack_neon_بھرو_f16x8(r2 + 2 * cp));
            ثابت uint32x4_t v3 = vreinterpretq_u32_u16(repack_neon_بھرو_f16x8(r3 + 2 * cp));

            ثابت uint32x2x2_t t0 = vtrn_u32(vget_کم_u32(v0),  vget_کم_u32(v1));
            ثابت uint32x2x2_t t1 = vtrn_u32(vget_بلند_u32(v0), vget_بلند_u32(v1));
            ثابت uint32x2x2_t t2 = vtrn_u32(vget_کم_u32(v2),  vget_کم_u32(v3));
            ثابت uint32x2x2_t t3 = vtrn_u32(vget_بلند_u32(v2), vget_بلند_u32(v3));

            vst1q_u16(tile_dst + (cp + 0) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t0.val[0], t2.val[0])));
            vst1q_u16(tile_dst + (cp + 1) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t0.val[1], t2.val[1])));
            vst1q_u16(tile_dst + (cp + 2) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t1.val[0], t3.val[0])));
            vst1q_u16(tile_dst + (cp + 3) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t1.val[1], t3.val[1])));
        }
    }
}
#اگر_ختم

// --- Generic 32x32 tile grid walkers for block-32 and float types ---

سانچہ <قسم_نام Fn>
ساکن خالی repack_parallel_for_ct(size_t قاش_بائٹیں, صحیح n_ستون_tiles, Fn && fn) {
    صحیح n_دھاگے = 1;
    اگر (قاش_بائٹیں > 128 * 1024) {
        برائے (صحیح nt : { 4, 3, 2 }) {
            اگر (n_ستون_tiles % nt == 0) {
                n_دھاگے = nt;
                توڑو;
            }
        }
    }

    اگر (n_دھاگے <= 1) {
        fn(0, n_ستون_tiles);
        واپس;
    }

    ثابت صحیح chunk = n_ستون_tiles / n_دھاگے;
    std::thread workers[3];
    برائے (صحیح i = 0; i < n_دھاگے - 1; ++i) {
        workers[i] = std::thread([&fn, i, chunk]() {
            fn(i * chunk, (i + 1) * chunk);
        });
    }
    fn((n_دھاگے - 1) * chunk, n_ستون_tiles);
    برائے (صحیح i = 0; i < n_دھاگے - 1; ++i) {
        workers[i].join();
    }
}

سانچہ <قسم_نام SrcElemT, صحیح ELEMS_فی_KTILE, قسم_نام PackTileFn>
ساکن خالی repack_مقصد_tiled_grid_32(ggml_tensor * t, ثابت خالی * ڈیٹا, size_t انحراف, size_t size,
                                    size_t tile_سائز, PackTileFn && pack_tile) {
    ثابت SrcElemT * src_میٹرکس = (ثابت SrcElemT *) ڈیٹا;
    ثابت int64_t ne0 = t->ne[0];
    ثابت int64_t ne1 = t->ne[1];
    ثابت int64_t ne2 = t->ne[2];
    ثابت int64_t ne3 = t->ne[3];
    ثابت صحیح n_ستون_tiles = hex_باری_اوپر(ne1, 32) / 32;
    ثابت صحیح n_k_tiles   = hex_باری_اوپر(ne0, 32) / 32;
    ثابت size_t میٹرکس_سائز = (size_t) n_ستون_tiles * n_k_tiles * tile_سائز;

    ثابت size_t قاش_سائز = (size_t) ne1 * ggml_صف_سائز(t->type, ne0);
    ثابت int64_t آغاز_قاش = انحراف / قاش_سائز;
    int64_t انتہا_قاش = (انحراف + size + قاش_سائز - 1) / قاش_سائز;
    اگر (انتہا_قاش > ne2 * ne3) {
        انتہا_قاش = ne2 * ne3;
    }

    ثابت int64_t صف_stride = (ne0 / (32 / ELEMS_فی_KTILE));

    برائے (int64_t قاش_idx = آغاز_قاش; قاش_idx < انتہا_قاش; قاش_idx++) {
        ثابت SrcElemT * src_قاش = src_میٹرکس + (قاش_idx - آغاز_قاش) * (ne1 * صف_stride);
        uint8_t * میٹرکس_dst = (uint8_t *) t->ڈیٹا + قاش_idx * میٹرکس_سائز;

        repack_parallel_for_ct(میٹرکس_سائز, n_ستون_tiles, [&](صحیح ct0, صحیح ct1) {
            برائے (صحیح ct = ct0; ct < ct1; ct++) {
                برائے (صحیح kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = میٹرکس_dst + ((size_t) ct * n_k_tiles + kt) * tile_سائز;
                    pack_tile(tile_dst, src_قاش, ne0, ne1, صف_stride, ct, kt);
                }
            }
        });
    }
}

سانچہ <قسم_نام DstElemT, صحیح ELEMS_فی_KTILE, قسم_نام UnpackTileFn>
ساکن خالی unpack_from_tiled_grid_32(خالی * ڈیٹا, ثابت ggml_tensor * t, size_t انحراف, size_t size,
                                      size_t tile_سائز, UnpackTileFn && unpack_tile) {
    uint8_t * dst_بائٹیں = (uint8_t *) ڈیٹا;
    ثابت int64_t ne0 = t->ne[0];
    ثابت int64_t ne1 = t->ne[1];
    ثابت int64_t ne2 = t->ne[2];
    ثابت int64_t ne3 = t->ne[3];
    ثابت صحیح n_ستون_tiles = hex_باری_اوپر(ne1, 32) / 32;
    ثابت صحیح n_k_tiles   = hex_باری_اوپر(ne0, 32) / 32;
    ثابت size_t میٹرکس_سائز = (size_t) n_ستون_tiles * n_k_tiles * tile_سائز;

    ثابت size_t صف_سائز_بائٹیں = ggml_صف_سائز(t->type, ne0);
    ثابت size_t قاش_سائز     = (size_t) ne1 * صف_سائز_بائٹیں;
    ثابت int64_t آغاز_قاش   = انحراف / قاش_سائز;
    int64_t انتہا_قاش           = (انحراف + size + قاش_سائز - 1) / قاش_سائز;
    اگر (انتہا_قاش > ne2 * ne3) {
        انتہا_قاش = ne2 * ne3;
    }

    ثابت int64_t صف_stride = (ne0 / (32 / ELEMS_فی_KTILE));

    برائے (int64_t قاش_idx = آغاز_قاش; قاش_idx < انتہا_قاش; قاش_idx++) {
        ثابت size_t cur_آغاز_byte     = (std::max)(انحراف, (size_t) قاش_idx * قاش_سائز);
        ثابت size_t cur_انتہا_byte       = (std::min)(انحراف + size, (size_t) (قاش_idx + 1) * قاش_سائز);
        ثابت size_t قاش_انحراف_آغاز = cur_آغاز_byte - (size_t) قاش_idx * قاش_سائز;
        ثابت size_t قاش_انحراف_انتہا   = cur_انتہا_byte - (size_t) قاش_idx * قاش_سائز;

        ثابت int64_t آغاز_صف = قاش_انحراف_آغاز / صف_سائز_بائٹیں;
        ثابت int64_t انتہا_صف   = (std::min)((int64_t) ((قاش_انحراف_انتہا + صف_سائز_بائٹیں - 1) / صف_سائز_بائٹیں), ne1);

        ثابت صحیح آغاز_ct = (صحیح) (آغاز_صف / 32);
        ثابت صحیح انتہا_ct   = (std::min)((صحیح) ((انتہا_صف + 31) / 32), n_ستون_tiles);

        DstElemT * dst_قاش = (DstElemT *) (dst_بائٹیں + (cur_آغاز_byte - انحراف));
        ثابت uint8_t * میٹرکس_src = (ثابت uint8_t *) t->ڈیٹا + قاش_idx * میٹرکس_سائز;

        برائے (صحیح ct = آغاز_ct; ct < انتہا_ct; ct++) {
            برائے (صحیح kt = 0; kt < n_k_tiles; kt++) {
                ثابت uint8_t * tile_src = میٹرکس_src + ((size_t) ct * n_k_tiles + kt) * tile_سائز;
                unpack_tile(dst_قاش, tile_src, ne0, آغاز_صف, انتہا_صف, صف_stride, ct, kt);
            }
        }
    }
}

// --- Block-32 Quantized Types (Q4_0, IQ4_NL, Q4_1, Q8_0, MXFP4) ---

ساکن خالی repack_q4_0_tiled(ggml_tensor * t, ثابت خالی * ڈیٹا, size_t انحراف, size_t size) {
    repack_مقصد_tiled_grid_32<بلاک_q4_0, 1>(t, ڈیٹا, انحراف, size, HTP_TENSOR_TILE_سائز_Q4_0,
        [](uint8_t * tile_dst, ثابت بلاک_q4_0 * src_قاش, int64_t ne0, int64_t ne1, int64_t صف_stride, صحیح ct, صحیح kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
#اگر defined(__ARM_NEON) || defined(__aarch64__)
            اگر (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                برائے (صحیح row0 = 0; row0 < 32; row0 += 8) {
                    ثابت uint8_t * qs[8];
                    برائے (صحیح rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_قاش[(ct * 32 + row0 + rr) * صف_stride + kt].qs;
                    }
                    repack_neon_4bit_صفیں(qs, tile_dst, row0);
                }

                برائے (صحیح صف = 0; صف < 32; ++صف) {
                    scale_dst[صف] = src_قاش[(ct * 32 + صف) * صف_stride + kt].d;
                }
                واپس;
            }
#اگر_ختم
            uint8_t tile_quants[32][32];
            برائے (صحیح صف = 0; صف < 32; صف++) {
                ثابت int64_t r = ct * 32 + صف;
                اگر (r < ne1 && kt < ne0 / 32) {
                    ثابت بلاک_q4_0 & b = src_قاش[r * صف_stride + kt];
                    unpack_nibbles_32(tile_quants[صف], b.qs);
                    scale_dst[صف] = b.d;
                } ورنہ {
                    memset(tile_quants[صف], 8, 32);
                    scale_dst[صف] = 0;
                }
            }
            transpose_nibbles_مقصد_tile(tile_dst, tile_quants);
        });
}

ساکن خالی repack_tiled_q4_0(خالی * ڈیٹا, ثابت ggml_tensor * t, size_t انحراف, size_t size) {
    unpack_from_tiled_grid_32<بلاک_q4_0, 1>(ڈیٹا, t, انحراف, size, HTP_TENSOR_TILE_سائز_Q4_0,
        [](بلاک_q4_0 * dst_قاش, ثابت uint8_t * tile_src, int64_t ne0, int64_t آغاز_صف, int64_t انتہا_صف, int64_t صف_stride, صحیح ct, صحیح kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_مقصد_nibbles(tile_quants, tile_src);
            ثابت ggml_half * scale_src = (ثابت ggml_half *) (tile_src + 512);
            برائے (صحیح صف = 0; صف < 32; صف++) {
                ثابت int64_t r = ct * 32 + صف;
                اگر (r >= آغاز_صف && r < انتہا_صف && kt < ne0 / 32) {
                    بلاک_q4_0 & b = dst_قاش[(r - آغاز_صف) * صف_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[صف]);
                    b.d = scale_src[صف];
                }
            }
        });
}

ساکن خالی repack_q4_1_tiled(ggml_tensor * t, ثابت خالی * ڈیٹا, size_t انحراف, size_t size) {
    repack_مقصد_tiled_grid_32<بلاک_q4_1, 1>(t, ڈیٹا, انحراف, size, HTP_TENSOR_TILE_سائز_Q4_1,
        [](uint8_t * tile_dst, ثابت بلاک_q4_1 * src_قاش, int64_t ne0, int64_t ne1, int64_t صف_stride, صحیح ct, صحیح kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
#اگر defined(__ARM_NEON) || defined(__aarch64__)
            اگر (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                برائے (صحیح row0 = 0; row0 < 32; row0 += 8) {
                    ثابت uint8_t * qs[8];
                    برائے (صحیح rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_قاش[(ct * 32 + row0 + rr) * صف_stride + kt].qs;
                    }
                    repack_neon_4bit_صفیں(qs, tile_dst, row0);
                }

                برائے (صحیح صف = 0; صف < 32; ++صف) {
                    ثابت بلاک_q4_1 & src_بلاک = src_قاش[(ct * 32 + صف) * صف_stride + kt];
                    scale_dst[2 * صف + 0] = src_بلاک.d;
                    scale_dst[2 * صف + 1] = src_بلاک.m;
                }
                واپس;
            }
#اگر_ختم
            uint8_t tile_quants[32][32];
            برائے (صحیح صف = 0; صف < 32; صف++) {
                ثابت int64_t r = ct * 32 + صف;
                اگر (r < ne1 && kt < ne0 / 32) {
                    ثابت بلاک_q4_1 & b = src_قاش[r * صف_stride + kt];
                    unpack_nibbles_32(tile_quants[صف], b.qs);
                    scale_dst[2 * صف + 0] = b.d;
                    scale_dst[2 * صف + 1] = b.m;
                } ورنہ {
                    memset(tile_quants[صف], 0, 32);
                    scale_dst[2 * صف + 0] = 0;
                    scale_dst[2 * صف + 1] = 0;
                }
            }
            transpose_nibbles_مقصد_tile(tile_dst, tile_quants);
        });
}

ساکن خالی repack_tiled_q4_1(خالی * ڈیٹا, ثابت ggml_tensor * t, size_t انحراف, size_t size) {
    unpack_from_tiled_grid_32<بلاک_q4_1, 1>(ڈیٹا, t, انحراف, size, HTP_TENSOR_TILE_سائز_Q4_1,
        [](بلاک_q4_1 * dst_قاش, ثابت uint8_t * tile_src, int64_t ne0, int64_t آغاز_صف, int64_t انتہا_صف, int64_t صف_stride, صحیح ct, صحیح kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_مقصد_nibbles(tile_quants, tile_src);
            ثابت ggml_half * scale_src = (ثابت ggml_half *) (tile_src + 512);
            برائے (صحیح صف = 0; صف < 32; صف++) {
                ثابت int64_t r = ct * 32 + صف;
                اگر (r >= آغاز_صف && r < انتہا_صف && kt < ne0 / 32) {
                    بلاک_q4_1 & b = dst_قاش[(r - آغاز_صف) * صف_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[صف]);
                    b.d = scale_src[2 * صف + 0];
                    b.m = scale_src[2 * صف + 1];
                }
            }
        });
}

ساکن خالی repack_mxfp4_tiled(ggml_tensor * t, ثابت خالی * ڈیٹا, size_t انحراف, size_t size) {
    repack_مقصد_tiled_grid_32<بلاک_mxfp4, 1>(t, ڈیٹا, انحراف, size, HTP_TENSOR_TILE_سائز_MXFP4,
        [](uint8_t * tile_dst, ثابت بلاک_mxfp4 * src_قاش, int64_t ne0, int64_t ne1, int64_t صف_stride, صحیح ct, صحیح kt) {
            uint8_t * scale_dst = tile_dst + 512;
#اگر defined(__ARM_NEON) || defined(__aarch64__)
            اگر (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                برائے (صحیح row0 = 0; row0 < 32; row0 += 8) {
                    ثابت uint8_t * qs[8];
                    برائے (صحیح rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_قاش[(ct * 32 + row0 + rr) * صف_stride + kt].qs;
                    }
                    repack_neon_4bit_صفیں(qs, tile_dst, row0);
                }

                برائے (صحیح صف = 0; صف < 32; ++صف) {
                    scale_dst[صف] = src_قاش[(ct * 32 + صف) * صف_stride + kt].e;
                }
                واپس;
            }
#اگر_ختم
            uint8_t tile_quants[32][32];
            برائے (صحیح صف = 0; صف < 32; صف++) {
                ثابت int64_t r = ct * 32 + صف;
                اگر (r < ne1 && kt < ne0 / 32) {
                    ثابت بلاک_mxfp4 & b = src_قاش[r * صف_stride + kt];
                    unpack_nibbles_32(tile_quants[صف], b.qs);
                    scale_dst[صف] = b.e;
                } ورنہ {
                    memset(tile_quants[صف], 0, 32);
                    scale_dst[صف] = 0;
                }
            }
            transpose_nibbles_مقصد_tile(tile_dst, tile_quants);
        });
}

ساکن خالی repack_tiled_mxfp4(خالی * ڈیٹا, ثابت ggml_tensor * t, size_t انحراف, size_t size) {
    unpack_from_tiled_grid_32<بلاک_mxfp4, 1>(ڈیٹا, t, انحراف, size, HTP_TENSOR_TILE_سائز_MXFP4,
        [](بلاک_mxfp4 * dst_قاش, ثابت uint8_t * tile_src, int64_t ne0, int64_t آغاز_صف, int64_t انتہا_صف, int64_t صف_stride, صحیح ct, صحیح kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_مقصد_nibbles(tile_quants, tile_src);
            ثابت uint8_t * scale_src = tile_src + 512;
            برائے (صحیح صف = 0; صف < 32; صف++) {
                ثابت int64_t r = ct * 32 + صف;
                اگر (r >= آغاز_صف && r < انتہا_صف && kt < ne0 / 32) {
                    بلاک_mxfp4 & b = dst_قاش[(r - آغاز_صف) * صف_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[صف]);
                    b.e = scale_src[صف];
                }
            }
        });
}

ساکن خالی repack_q8_0_tiled(ggml_tensor * t, ثابت خالی * ڈیٹا, size_t انحراف, size_t size) {
    repack_مقصد_tiled_grid_32<بلاک_q8_0, 1>(t, ڈیٹا, انحراف, size, HTP_TENSOR_TILE_سائز_Q8_0,
        [](uint8_t * tile_dst, ثابت بلاک_q8_0 * src_قاش, int64_t ne0, int64_t ne1, int64_t صف_stride, صحیح ct, صحیح kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 1024);
#اگر defined(__ARM_NEON) || defined(__aarch64__)
            اگر (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                برائے (صحیح row0 = 0; row0 < 32; row0 += 8) {
                    ثابت uint8_t * qs[8];
                    برائے (صحیح rr = 0; rr < 8; ++rr) {
                        qs[rr] = (ثابت uint8_t *) src_قاش[(ct * 32 + row0 + rr) * صف_stride + kt].qs;
                    }
                    repack_neon_q8_صفیں(qs, tile_dst, row0);
                }

                برائے (صحیح صف = 0; صف < 32; ++صف) {
                    scale_dst[صف] = src_قاش[(ct * 32 + صف) * صف_stride + kt].d;
                }
                واپس;
            }
#اگر_ختم
            برائے (صحیح cp = 0; cp < 16; cp++) {
                ثابت صحیح col0 = cp * 2;
                ثابت صحیح col1 = col0 + 1;
                برائے (صحیح صف = 0; صف < 32; صف++) {
                    ثابت int64_t r = ct * 32 + صف;
                    ثابت بلاک_q8_0 * b = (r < ne1 && kt < ne0 / 32) ? &src_قاش[r * صف_stride + kt] : خالی_اشاریہ;
                    tile_dst[cp * 64 + 2 * صف + 0] = b ? b->qs[col0] : 0;
                    tile_dst[cp * 64 + 2 * صف + 1] = b ? b->qs[col1] : 0;
                }
            }
            برائے (صحیح صف = 0; صف < 32; صف++) {
                ثابت int64_t r = ct * 32 + صف;
                scale_dst[صف] = (r < ne1 && kt < ne0 / 32) ? src_قاش[r * صف_stride + kt].d : 0;
            }
        });
}

ساکن خالی repack_tiled_q8_0(خالی * ڈیٹا, ثابت ggml_tensor * t, size_t انحراف, size_t size) {
    unpack_from_tiled_grid_32<بلاک_q8_0, 1>(ڈیٹا, t, انحراف, size, HTP_TENSOR_TILE_سائز_Q8_0,
        [](بلاک_q8_0 * dst_قاش, ثابت uint8_t * tile_src, int64_t ne0, int64_t آغاز_صف, int64_t انتہا_صف, int64_t صف_stride, صحیح ct, صحیح kt) {
            برائے (صحیح cp = 0; cp < 16; cp++) {
                ثابت صحیح col0 = cp * 2;
                ثابت صحیح col1 = col0 + 1;
                برائے (صحیح صف = 0; صف < 32; صف++) {
                    ثابت int64_t r = ct * 32 + صف;
                    اگر (r >= آغاز_صف && r < انتہا_صف && kt < ne0 / 32) {
                        بلاک_q8_0 & b = dst_قاش[(r - آغاز_صف) * صف_stride + kt];
                        b.qs[col0] = tile_src[cp * 64 + 2 * صف + 0];
                        b.qs[col1] = tile_src[cp * 64 + 2 * صف + 1];
                    }
                }
            }
            ثابت ggml_half * scale_src = (ثابت ggml_half *) (tile_src + 1024);
            برائے (صحیح صف = 0; صف < 32; صف++) {
                ثابت int64_t r = ct * 32 + صف;
                اگر (r >= آغاز_صف && r < انتہا_صف && kt < ne0 / 32) {
                    dst_قاش[(r - آغاز_صف) * صف_stride + kt].d = scale_src[صف];
                }
            }
        });
}

// --- Float Types (F16, F32, BF16 -> f16_tiled HMX layout) ---

ساکن خالی repack_q2_0_tiled(ggml_tensor * t, ثابت خالی * ڈیٹا, size_t انحراف, size_t size) {
    GGML_ASSERT(انحراف == 0);
    GGML_ASSERT(t->ne[0] % 64 == 0);
    GGML_ASSERT(size == ggml_nbytes(t));

    ثابت بلاک_q2_0 * src = (ثابت بلاک_q2_0 *) ڈیٹا;
    ثابت int64_t بلاکیں_فی_صف = t->ne[0] / 64;
    ثابت int64_t n_ستون_tiles = hex_باری_اوپر(t->ne[1], 32) / 32;
    ثابت int64_t n_وزن_tiles = بلاکیں_فی_صف;
    ثابت size_t میٹرکس_سائز = (size_t) n_ستون_tiles * n_وزن_tiles * HTP_TENSOR_TILE_سائز_Q2_0;

    برائے (int64_t قاش_2 = 0; قاش_2 < t->ne[2] * t->ne[3]; ++قاش_2) {
        ثابت بلاک_q2_0 * src_قاش = src + قاش_2 * t->ne[1] * بلاکیں_فی_صف;
        uint8_t * dst_قاش = (uint8_t *) t->ڈیٹا + قاش_2 * میٹرکس_سائز;
        ثابت size_t ct_بائٹیں = (size_t) n_وزن_tiles * HTP_TENSOR_TILE_سائز_Q2_0;

        repack_parallel_for_ct(میٹرکس_سائز, (صحیح) n_ستون_tiles, [&](صحیح ct0, صحیح ct1) {
            memset(dst_قاش + (size_t) ct0 * ct_بائٹیں, 0, (size_t) (ct1 - ct0) * ct_بائٹیں);
            ثابت int64_t r0 = (int64_t) ct0 * 32;
            ثابت int64_t r1 = (std::min)((int64_t) ct1 * 32, t->ne[1]);
            برائے (int64_t صف_اشاریہ = r0; صف_اشاریہ < r1; ++صف_اشاریہ) {
                ثابت int64_t ct = صف_اشاریہ / 32;
                ثابت صحیح صف = (صحیح) (صف_اشاریہ % 32);
                برائے (int64_t kt = 0; kt < n_وزن_tiles; ++kt) {
                    ثابت بلاک_q2_0 * بلاک = src_قاش + صف_اشاریہ * بلاکیں_فی_صف + kt;
                    uint8_t * tile = dst_قاش + (ct * n_وزن_tiles + kt) * HTP_TENSOR_TILE_سائز_Q2_0;
#اگر defined(__ARM_NEON) || defined(__aarch64__)
                    repack_neon_q2_0_بلاک(بلاک->qs, tile, صف);
#ورنہ
                    برائے (صحیح گروہ = 0; گروہ < 16; ++گروہ) {
                        برائے (صحیح byte = 0; byte < 4; ++byte) {
                            ثابت صحیح k = 4 * گروہ + byte;
                            ثابت uint8_t code = (بلاک->qs[k / 4] >> (2 * (k % 4))) & 3;
                            tile[(گروہ >> 2) * 128 + 4 * صف + byte] |= code << (2 * (گروہ & 3));
                        }
                    }
#اگر_ختم
                    حافظہ_نقل(tile + 512 + 2 * صف, &بلاک->d, حجم(بلاک->d));
                }
            }
        });
    }
}

ساکن خالی repack_tiled_q2_0(خالی * ڈیٹا, ثابت ggml_tensor * t, size_t انحراف, size_t size) {
    GGML_ASSERT(t->ne[0] % 64 == 0);
    GGML_ASSERT(انحراف + size <= ggml_nbytes(t));

    ثابت size_t پہلا_بلاک = انحراف / حجم(بلاک_q2_0);
    ثابت size_t انتہا_بلاک = (انحراف + size + حجم(بلاک_q2_0) - 1) / حجم(بلاک_q2_0);
    ثابت size_t بلاکیں_فی_صف = t->ne[0] / 64;
    ثابت size_t بلاکیں_فی_قاش = t->ne[1] * بلاکیں_فی_صف;
    ثابت size_t n_وزن_tiles = بلاکیں_فی_صف;
    ثابت size_t n_ستون_tiles = hex_باری_اوپر(t->ne[1], 32) / 32;
    ثابت size_t میٹرکس_سائز = n_ستون_tiles * n_وزن_tiles * HTP_TENSOR_TILE_سائز_Q2_0;

    برائے (size_t بلاک_اشاریہ = پہلا_بلاک; بلاک_اشاریہ < انتہا_بلاک; ++بلاک_اشاریہ) {
        ثابت size_t قاش_2 = بلاک_اشاریہ / بلاکیں_فی_قاش;
        ثابت size_t اندر_قاش = بلاک_اشاریہ % بلاکیں_فی_قاش;
        ثابت size_t صف_اشاریہ = اندر_قاش / بلاکیں_فی_صف;
        ثابت size_t kt = اندر_قاش % بلاکیں_فی_صف;
        ثابت size_t ct = صف_اشاریہ / 32;
        ثابت size_t صف = صف_اشاریہ % 32;
        ثابت uint8_t * tile = (ثابت uint8_t *) t->ڈیٹا + قاش_2 * میٹرکس_سائز +
            (ct * n_وزن_tiles + kt) * HTP_TENSOR_TILE_سائز_Q2_0;

        بلاک_q2_0 بلاک = {};
        حافظہ_نقل(&بلاک.d, tile + 512 + 2 * صف, حجم(بلاک.d));
        برائے (صحیح گروہ = 0; گروہ < 16; ++گروہ) {
            برائے (صحیح byte = 0; byte < 4; ++byte) {
                ثابت uint8_t packed = tile[(گروہ >> 2) * 128 + 4 * صف + byte];
                ثابت uint8_t code = (packed >> (2 * (گروہ & 3))) & 3;
                بلاک.qs[گروہ] |= code << (2 * byte);
            }
        }

        ثابت size_t بلاک_begin = بلاک_اشاریہ * حجم(بلاک_q2_0);
        ثابت size_t copy_begin = (std::max)(بلاک_begin, انحراف);
        ثابت size_t copy_انتہا = (std::min)(بلاک_begin + حجم(بلاک_q2_0), انحراف + size);
        حافظہ_نقل((uint8_t *) ڈیٹا + copy_begin - انحراف, (ثابت uint8_t *) &بلاک + copy_begin - بلاک_begin, copy_انتہا - copy_begin);
    }
}

سانچہ <قسم_نام SrcT, قسم_نام مقصد_Fp16_Fn>
ساکن خالی repack_float_مقصد_f16_tiled(ggml_tensor * t, ثابت خالی * ڈیٹا, size_t انحراف, size_t size, مقصد_Fp16_Fn && مقصد_fp16) {
    repack_مقصد_tiled_grid_32<SrcT, 32>(t, ڈیٹا, انحراف, size, HTP_TENSOR_TILE_سائز_F16,
        [&](uint8_t * tile_بائٹیں, ثابت SrcT * src_قاش, int64_t ne0, int64_t ne1, int64_t صف_stride, صحیح ct, صحیح kt) {
            uint16_t * tile_dst = (uint16_t *) tile_بائٹیں;
#اگر defined(__ARM_NEON) || defined(__aarch64__)
            اگر (ct * 32 + 32 <= ne1 && kt * 32 + 32 <= ne0) {
                repack_neon_f16_tile(tile_dst, src_قاش, صف_stride, ct, kt);
                واپس;
            }
#اگر_ختم
            برائے (صحیح cp = 0; cp < 16; cp++) {
                ثابت int64_t c0 = kt * 32 + cp * 2;
                ثابت int64_t c1 = c0 + 1;
                برائے (صحیح صف = 0; صف < 32; صف++) {
                    ثابت int64_t r = ct * 32 + صف;
                    ثابت SrcT * r_ptr = (r < ne1) ? &src_قاش[r * صف_stride] : خالی_اشاریہ;
                    tile_dst[cp * 64 + صف * 2 + 0] = (r_ptr && c0 < ne0) ? مقصد_fp16(r_ptr[c0]) : 0;
                    tile_dst[cp * 64 + صف * 2 + 1] = (r_ptr && c1 < ne0) ? مقصد_fp16(r_ptr[c1]) : 0;
                }
            }
        });
}

سانچہ <قسم_نام DstT, قسم_نام FromFp16Fn>
ساکن خالی unpack_f16_tiled_مقصد_float(خالی * ڈیٹا, ثابت ggml_tensor * t, size_t انحراف, size_t size, FromFp16Fn && from_fp16) {
    unpack_from_tiled_grid_32<DstT, 32>(ڈیٹا, t, انحراف, size, HTP_TENSOR_TILE_سائز_F16,
        [&](DstT * dst_قاش, ثابت uint8_t * tile_بائٹیں, int64_t ne0, int64_t آغاز_صف, int64_t انتہا_صف, int64_t صف_stride, صحیح ct, صحیح kt) {
            ثابت uint16_t * tile_src = (ثابت uint16_t *) tile_بائٹیں;
            برائے (صحیح cp = 0; cp < 16; cp++) {
                ثابت int64_t c0 = kt * 32 + cp * 2;
                ثابت int64_t c1 = c0 + 1;
                برائے (صحیح صف = 0; صف < 32; صف++) {
                    ثابت int64_t r = ct * 32 + صف;
                    اگر (r >= آغاز_صف && r < انتہا_صف) {
                        DstT * r_ptr = dst_قاش + (r - آغاز_صف) * صف_stride;
                        اگر (c0 < ne0) r_ptr[c0] = from_fp16(tile_src[cp * 64 + صف * 2 + 0]);
                        اگر (c1 < ne0) r_ptr[c1] = from_fp16(tile_src[cp * 64 + صف * 2 + 1]);
                    }
                }
            }
        });
}

// --- Generic K-Quant (super-block 256) grid walkers ---

سانچہ <قسم_نام بلاک_T, قسم_نام PackKtFn>
ساکن خالی repack_kquant_مقصد_tiled_grid(ggml_tensor * t, ثابت خالی * ڈیٹا, size_t انحراف, size_t size,
                                        size_t tile_سائز, PackKtFn && pack_kt) {
    GGML_ASSERT(انحراف == 0);
    GGML_غیر_مستعمل(size);

    ثابت بلاک_T * src_میٹرکس = (ثابت بلاک_T *) ڈیٹا;
    ثابت int64_t ne0 = t->ne[0];
    ثابت int64_t ne1 = t->ne[1];
    ثابت int64_t ne2 = t->ne[2];
    ثابت int64_t ne3 = t->ne[3];
    GGML_ASSERT(ne0 % QK_K == 0);

    ثابت صحیح n_ستون_tiles    = hex_باری_اوپر(ne1, 32) / 32;
    ثابت صحیح n_k_tiles      = hex_باری_اوپر(ne0, 32) / 32;
    ثابت size_t میٹرکس_سائز = (size_t) n_ستون_tiles * n_k_tiles * tile_سائز;
    ثابت int64_t sb_فی_صف = ne0 / QK_K;

    برائے (int64_t قاش_idx = 0; قاش_idx < ne2 * ne3; قاش_idx++) {
        ثابت بلاک_T * src_قاش = src_میٹرکس + قاش_idx * (ne1 * sb_فی_صف);
        uint8_t * میٹرکس_dst     = (uint8_t *) t->ڈیٹا + قاش_idx * میٹرکس_سائز;
        ثابت size_t ct_بائٹیں    = (size_t) n_k_tiles * tile_سائز;

        repack_parallel_for_ct(میٹرکس_سائز, n_ستون_tiles, [&](صحیح ct0, صحیح ct1) {
            // Padding rows and OR-ed bit planes require zeroed tiles
            memset(میٹرکس_dst + (size_t) ct0 * ct_بائٹیں, 0, (size_t) (ct1 - ct0) * ct_بائٹیں);

            ثابت int64_t r0 = (int64_t) ct0 * 32;
            ثابت int64_t r1 = (std::min)((int64_t) ct1 * 32, ne1);
            برائے (int64_t r = r0; r < r1; r++) {
                ثابت صحیح ct  = (صحیح) (r / 32);
                ثابت صحیح صف = (صحیح) (r % 32);
                ثابت بلاک_T * src_صف = src_قاش + r * sb_فی_صف;

                برائے (صحیح kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile = میٹرکس_dst + ((size_t) ct * n_k_tiles + kt) * tile_سائز;
                    pack_kt(tile, &src_صف[kt / 8], صف, kt % 8);
                }
            }
        });
    }
}

سانچہ <قسم_نام بلاک_T, قسم_نام UnpackSbFn>
ساکن خالی unpack_kquant_from_tiled_grid(خالی * ڈیٹا, ثابت ggml_tensor * t, size_t انحراف, size_t size,
                                          size_t tile_سائز, UnpackSbFn && unpack_sb) {
    uint8_t * dst_بائٹیں = (uint8_t *) ڈیٹا;
    ثابت int64_t ne0 = t->ne[0];
    ثابت int64_t ne1 = t->ne[1];
    ثابت int64_t ne2 = t->ne[2];
    ثابت int64_t ne3 = t->ne[3];
    GGML_ASSERT(ne0 % QK_K == 0);

    ثابت صحیح n_ستون_tiles    = hex_باری_اوپر(ne1, 32) / 32;
    ثابت صحیح n_k_tiles      = hex_باری_اوپر(ne0, 32) / 32;
    ثابت size_t میٹرکس_سائز = (size_t) n_ستون_tiles * n_k_tiles * tile_سائز;
    ثابت int64_t sb_فی_صف = ne0 / QK_K;

    ثابت size_t صف_سائز_بائٹیں = ggml_صف_سائز(t->type, ne0);
    ثابت size_t قاش_سائز     = (size_t) ne1 * صف_سائز_بائٹیں;
    ثابت int64_t آغاز_قاش   = انحراف / قاش_سائز;
    int64_t انتہا_قاش           = (انحراف + size + قاش_سائز - 1) / قاش_سائز;
    اگر (انتہا_قاش > ne2 * ne3) {
        انتہا_قاش = ne2 * ne3;
    }

    برائے (int64_t قاش_idx = آغاز_قاش; قاش_idx < انتہا_قاش; قاش_idx++) {
        ثابت size_t cur_آغاز_byte     = (std::max)(انحراف, (size_t) قاش_idx * قاش_سائز);
        ثابت size_t cur_انتہا_byte       = (std::min)(انحراف + size, (size_t) (قاش_idx + 1) * قاش_سائز);
        ثابت size_t قاش_انحراف_آغاز = cur_آغاز_byte - (size_t) قاش_idx * قاش_سائز;
        ثابت size_t قاش_انحراف_انتہا   = cur_انتہا_byte - (size_t) قاش_idx * قاش_سائز;

        ثابت int64_t آغاز_صف = قاش_انحراف_آغاز / صف_سائز_بائٹیں;
        ثابت int64_t انتہا_صف   = (std::min)((int64_t) ((قاش_انحراف_انتہا + صف_سائز_بائٹیں - 1) / صف_سائز_بائٹیں), ne1);

        بلاک_T * dst_قاش         = (بلاک_T *) (dst_بائٹیں + (cur_آغاز_byte - انحراف));
        ثابت uint8_t * میٹرکس_src = (ثابت uint8_t *) t->ڈیٹا + قاش_idx * میٹرکس_سائز;

        برائے (int64_t r = آغاز_صف; r < انتہا_صف; r++) {
            ثابت صحیح ct  = (صحیح) (r / 32);
            ثابت صحیح صف = (صحیح) (r % 32);
            بلاک_T * dst_صف = dst_قاش + (r - آغاز_صف) * sb_فی_صف;

            برائے (int64_t sb = 0; sb < sb_فی_صف; sb++) {
                بلاک_T * b = &dst_صف[sb];
                memset(b, 0, حجم(بلاک_T));
                ثابت uint8_t * sb_tiles = میٹرکس_src + ((size_t) ct * n_k_tiles + sb * 8) * tile_سائز;
                unpack_sb(b, sb_tiles, tile_سائز, صف);
            }
        }
    }
}

// --- K-Quant Helpers & Per-Type Implementations (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) ---

#اگر !defined(__ARM_NEON) && !defined(__aarch64__)
// unsigned 6-bit value (0..63) of element e of a Q6_K block, same bit layout as dequantize_row_q6_K
ساکن اندرونی uint8_t q6_K_get_quant(ثابت بلاک_q6_K * b, صحیح e) {
    ثابت صحیح c = e / 128;
    ثابت صحیح w = e % 128;
    ثابت صحیح g = w / 32;
    ثابت صحیح l = w % 32;
    ثابت uint8_t * ql = b->ql + c * 64;
    ثابت uint8_t * qh = b->qh + c * 32;
    uint8_t lo, hi;
    انتخاب (g) {
        حالت 0:  lo = ql[l]      & 0xF; hi = (qh[l] >> 0) & 3; توڑو;
        حالت 1:  lo = ql[l + 32] & 0xF; hi = (qh[l] >> 2) & 3; توڑو;
        حالت 2:  lo = ql[l]      >> 4;  hi = (qh[l] >> 4) & 3; توڑو;
        طےشدہ: lo = ql[l + 32] >> 4;  hi = (qh[l] >> 6) & 3; توڑو;
    }
    واپس (uint8_t) (lo | (hi << 4));
}
#اگر_ختم

ساکن خالی repack_q6_K_tiled(ggml_tensor * t, ثابت خالی * ڈیٹا, size_t انحراف, size_t size) {
    repack_kquant_مقصد_tiled_grid<بلاک_q6_K>(t, ڈیٹا, انحراف, size, HTP_TENSOR_TILE_سائز_Q6_K,
        [](uint8_t * tile, ثابت بلاک_q6_K * b, صحیح صف, صحیح kt_local) {
            ثابت اعشاری d     = GGML_FP16_مقصد_FP32(b->d);
            uint8_t * lo_pl   = tile;
            uint8_t * hi_pl   = tile + 512;
            ggml_half * sc_pl = (ggml_half *) (tile + 768);

#اگر defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q6_k(b, kt_local, lo_pl, hi_pl, صف);
#ورنہ
            برائے (صحیح lk = 0; lk < 32; lk++) {
                ثابت uint8_t q6 = q6_K_get_quant(b, kt_local * 32 + lk);
                ثابت صحیح g   = lk >> 2;
                ثابت صحیح pos = صف * 4 + (lk & 3);
                lo_pl[(g >> 1) * 128 + pos] |= (uint8_t) ((q6 & 0xF) << ((g & 1) * 4));
                hi_pl[(g >> 2) * 128 + pos] |= (uint8_t) ((q6 >> 4) << ((g & 3) * 2));
            }
#اگر_ختم
            برائے (صحیح sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + صف] = GGML_FP32_مقصد_FP16(d * (اعشاری) b->scales[kt_local * 2 + sub]);
            }
        });
}

ساکن خالی repack_tiled_q6_K(خالی * ڈیٹا, ثابت ggml_tensor * t, size_t انحراف, size_t size) {
    unpack_kquant_from_tiled_grid<بلاک_q6_K>(ڈیٹا, t, انحراف, size, HTP_TENSOR_TILE_سائز_Q6_K,
        [](بلاک_q6_K * b, ثابت uint8_t * sb_tiles, size_t tile_سائز, صحیح صف) {
            اعشاری sub_scales[16];
            برائے (صحیح kt_local = 0; kt_local < 8; kt_local++) {
                ثابت uint8_t *   tile  = sb_tiles + kt_local * tile_سائز;
                ثابت uint8_t *   lo_pl = tile;
                ثابت uint8_t *   hi_pl = tile + 512;
                ثابت ggml_half * sc_pl = (ثابت ggml_half *) (tile + 768);

                ثابت صحیح c  = kt_local / 4;
                ثابت صحیح g  = kt_local % 4;
                uint8_t * ql = b->ql + c * 64;
                uint8_t * qh = b->qh + c * 32;

                برائے (صحیح lk = 0; lk < 32; lk++) {
                    ثابت صحیح g_tile = lk >> 2;
                    ثابت صحیح pos    = صف * 4 + (lk & 3);
                    ثابت uint8_t lo = (lo_pl[(g_tile >> 1) * 128 + pos] >> ((g_tile & 1) * 4)) & 0xF;
                    ثابت uint8_t hi = (hi_pl[(g_tile >> 2) * 128 + pos] >> ((g_tile & 3) * 2)) & 3;

                    انتخاب (g) {
                        حالت 0:  ql[lk]      |= lo;        qh[lk] |= (hi << 0); توڑو;
                        حالت 1:  ql[lk + 32] |= lo;        qh[lk] |= (hi << 2); توڑو;
                        حالت 2:  ql[lk]      |= (lo << 4); qh[lk] |= (hi << 4); توڑو;
                        طےشدہ: ql[lk + 32] |= (lo << 4); qh[lk] |= (hi << 6); توڑو;
                    }
                }

                برائے (صحیح sub = 0; sub < 2; sub++) {
                    sub_scales[kt_local * 2 + sub] = GGML_FP16_مقصد_FP32(sc_pl[sub * 32 + صف]);
                }
            }

            اعشاری max_abs_scale = 0.0f;
            برائے (صحیح s = 0; s < 16; s++) {
                max_abs_scale = (std::max)(max_abs_scale, fabsf(sub_scales[s]));
            }

            اگر (max_abs_scale == 0.0f) {
                b->d = GGML_FP32_مقصد_FP16(0.0f);
            } ورنہ {
                b->d = GGML_FP32_مقصد_FP16(max_abs_scale / 127.0f);
                ثابت اعشاری d_actual = GGML_FP16_مقصد_FP32(b->d);
                ثابت اعشاری inv_d    = (d_actual != 0.0f) ? (1.0f / d_actual) : 0.0f;
                برائے (صحیح s = 0; s < 16; s++) {
                    ثابت صحیح sc = (صحیح) roundf(sub_scales[s] * inv_d);
                    b->scales[s] = (int8_t) (std::max)(-128, (std::min)(127, sc));
                }
            }
        });
}

#اگر !defined(__ARM_NEON) && !defined(__aarch64__)
// low 2 bits (0..3) of element e of a Q2_K or Q3_K block, same bit layout as dequantize_row_q2_K / q3_K
ساکن اندرونی uint8_t q2_3_K_get_low2(ثابت uint8_t * qs, صحیح e) {
    ثابت صحیح c = e / 128;
    ثابت صحیح j = (e % 128) / 32;
    ثابت صحیح l = e % 32;
    واپس (qs[c * 32 + l] >> (2 * j)) & 3;
}

// hmask bit of element e of a Q3_K block, same bit layout as dequantize_row_q3_K
ساکن اندرونی منطقی q3_K_get_hbit(ثابت بلاک_q3_K * b, صحیح e) {
    ثابت صحیح c = e / 128;
    ثابت صحیح j = (e % 128) / 32;
    ثابت صحیح l = e % 32;
    واپس (b->hmask[l] >> (c * 4 + j)) & 1;
}
#اگر_ختم

// signed 6-bit scale j (-32..31) of a Q3_K block, same packing as quantize_row_q3_K_ref
ساکن اندرونی صحیح q3_K_get_scale(ثابت uint8_t * scales, صحیح j) {
    ثابت صحیح lo = (j < 8) ? (scales[j] & 0xF) : (scales[j - 8] >> 4);
    ثابت صحیح hi = (scales[8 + j % 4] >> (2 * (j / 4))) & 3;
    واپس (lo | (hi << 4)) - 32;
}

// read-back: find fp16 d and l[j] in [lmin, lmax] with fp16(d * l[j]) == prod[j] for all j, false if none
ساکن منطقی hexagon_recover_k_scales(ثابت ggml_half * prod, صحیح n, صحیح lmin, صحیح lmax, ggml_half * d_برآمد, صحیح * l_برآمد) {
    صحیح jmax = 0;
    برائے (صحیح j = 1; j < n; j++) {
        اگر (fabsf(GGML_FP16_مقصد_FP32(prod[j])) > fabsf(GGML_FP16_مقصد_FP32(prod[jmax]))) {
            jmax = j;
        }
    }
    ثابت اعشاری pmax = GGML_FP16_مقصد_FP32(prod[jmax]);
    اگر (pmax == 0.0f) {
        *d_برآمد = GGML_FP32_مقصد_FP16(0.0f);
        برائے (صحیح j = 0; j < n; j++) {
            l_برآمد[j] = 0;
        }
        واپس سچ;
    }

    // the quantizers put the largest scale at or near the range end, so try large |l| first
    ثابت صحیح lext = (std::max)(-lmin, lmax);
    برائے (صحیح a = lext; a >= 1; a--) {
        برائے (صحیح sign : { 1, -1 }) {
            ثابت صحیح lj = sign * a;
            اگر (lj < lmin || lj > lmax) {
                جاری;
            }
            ثابت ggml_half d0 = GGML_FP32_مقصد_FP16(pmax / (اعشاری) lj);
            برائے (صحیح ulp : { 0, -1, 1 }) {
                ggml_half d = d0;
                uint16_t بٹ;
                حافظہ_نقل(&بٹ, &d, حجم(بٹ));
                بٹ = (uint16_t) (بٹ + ulp);
                حافظہ_نقل(&d, &بٹ, حجم(بٹ));

                ثابت اعشاری df = GGML_FP16_مقصد_FP32(d);
                اگر (!std::isfinite(df) || df == 0.0f) {
                    جاری;
                }
                منطقی ok = سچ;
                برائے (صحیح j = 0; j < n && ok; j++) {
                    ثابت صحیح l = (صحیح) roundf(GGML_FP16_مقصد_FP32(prod[j]) / df);
                    ثابت ggml_half p = GGML_FP32_مقصد_FP16(df * (اعشاری) l);
                    ok = l >= lmin && l <= lmax && memcmp(&p, &prod[j], حجم(p)) == 0;
                    l_برآمد[j] = l;
                }
                اگر (ok) {
                    *d_برآمد = d;
                    واپس سچ;
                }
            }
        }
    }
    واپس جھوٹ;
}

ساکن خالی repack_q3_K_tiled(ggml_tensor * t, ثابت خالی * ڈیٹا, size_t انحراف, size_t size) {
    repack_kquant_مقصد_tiled_grid<بلاک_q3_K>(t, ڈیٹا, انحراف, size, HTP_TENSOR_TILE_سائز_Q3_K,
        [](uint8_t * tile, ثابت بلاک_q3_K * b, صحیح صف, صحیح kt_local) {
            ثابت اعشاری d     = GGML_FP16_مقصد_FP32(b->d);
            uint8_t * lo_pl   = tile;
            uint8_t * neg_pl  = tile + 256;
            ggml_half * sc_pl = (ggml_half *) (tile + 384);

#اگر defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q2_3_k_low2(b->qs, kt_local, lo_pl, صف);
            repack_neon_q3_k_hmask(b->hmask, kt_local, neg_pl, صف);
#ورنہ
            برائے (صحیح lk = 0; lk < 32; lk++) {
                ثابت صحیح e   = kt_local * 32 + lk;
                ثابت صحیح g   = lk >> 2;
                ثابت صحیح pos = صف * 4 + (lk & 3);
                lo_pl[(g >> 2) * 128 + pos] |= (uint8_t) (q2_3_K_get_low2(b->qs, e) << ((g & 3) * 2));
                اگر (!q3_K_get_hbit(b, e)) {
                    neg_pl[pos] |= (uint8_t) (1 << g);
                }
            }
#اگر_ختم
            برائے (صحیح sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + صف] = GGML_FP32_مقصد_FP16(d * (اعشاری) q3_K_get_scale(b->scales, kt_local * 2 + sub));
            }
        });
}

ساکن خالی repack_tiled_q3_K(خالی * ڈیٹا, ثابت ggml_tensor * t, size_t انحراف, size_t size) {
    unpack_kquant_from_tiled_grid<بلاک_q3_K>(ڈیٹا, t, انحراف, size, HTP_TENSOR_TILE_سائز_Q3_K,
        [](بلاک_q3_K * b, ثابت uint8_t * sb_tiles, size_t tile_سائز, صحیح صف) {
            ggml_half sub_scales[16];
            برائے (صحیح kt_local = 0; kt_local < 8; kt_local++) {
                ثابت uint8_t *   tile   = sb_tiles + kt_local * tile_سائز;
                ثابت uint8_t *   lo_pl  = tile;
                ثابت uint8_t *   neg_pl = tile + 256;
                ثابت ggml_half * sc_pl  = (ثابت ggml_half *) (tile + 384);

                برائے (صحیح lk = 0; lk < 32; lk++) {
                    ثابت صحیح e   = kt_local * 32 + lk;
                    ثابت صحیح g   = lk >> 2;
                    ثابت صحیح pos = صف * 4 + (lk & 3);
                    ثابت uint8_t lo = (lo_pl[(g >> 2) * 128 + pos] >> ((g & 3) * 2)) & 3;

                    ثابت صحیح c = e / 128;
                    ثابت صحیح j = (e % 128) / 32;
                    ثابت صحیح l = e % 32;
                    b->qs[c * 32 + l] |= (uint8_t) (lo << (2 * j));
                    اگر (!((neg_pl[pos] >> g) & 1)) {
                        b->hmask[l] |= (uint8_t) (1 << (c * 4 + j));
                    }
                }

                برائے (صحیح sub = 0; sub < 2; sub++) {
                    sub_scales[kt_local * 2 + sub] = sc_pl[sub * 32 + صف];
                }
            }

            صحیح ls[16];
            اگر (!hexagon_recover_k_scales(sub_scales, 16, -32, 31, &b->d, ls)) {
                اعشاری max_scale = 0.0f;
                برائے (صحیح s = 0; s < 16; s++) {
                    اگر (fabsf(GGML_FP16_مقصد_FP32(sub_scales[s])) > fabsf(max_scale)) {
                        max_scale = GGML_FP16_مقصد_FP32(sub_scales[s]);
                    }
                }
                b->d = GGML_FP32_مقصد_FP16(-max_scale / 32.0f);
                ثابت اعشاری d_actual = GGML_FP16_مقصد_FP32(b->d);
                ثابت اعشاری inv_d    = (d_actual != 0.0f) ? (1.0f / d_actual) : 0.0f;
                برائے (صحیح s = 0; s < 16; s++) {
                    ls[s] = (std::max)(-32, (std::min)(31, (صحیح) roundf(GGML_FP16_مقصد_FP32(sub_scales[s]) * inv_d)));
                }
            }

            برائے (صحیح s = 0; s < 16; s++) {
                ثابت صحیح l = ls[s] + 32;
                اگر (s < 8) {
                    b->scales[s] = l & 0xF;
                } ورنہ {
                    b->scales[s - 8] |= (uint8_t) ((l & 0xF) << 4);
                }
                b->scales[s % 4 + 8] |= (uint8_t) ((l >> 4) << (2 * (s / 4)));
            }
        });
}

ساکن خالی repack_q2_K_tiled(ggml_tensor * t, ثابت خالی * ڈیٹا, size_t انحراف, size_t size) {
    repack_kquant_مقصد_tiled_grid<بلاک_q2_K>(t, ڈیٹا, انحراف, size, HTP_TENSOR_TILE_سائز_Q2_K,
        [](uint8_t * tile, ثابت بلاک_q2_K * b, صحیح صف, صحیح kt_local) {
            uint8_t * lo_pl   = tile;
            uint8_t * sc_pl   = tile + 256;
            ggml_half * dd_pl = (ggml_half *) (tile + 320);

#اگر defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q2_3_k_low2(b->qs, kt_local, lo_pl, صف);
#ورنہ
            برائے (صحیح lk = 0; lk < 32; lk++) {
                ثابت صحیح g   = lk >> 2;
                ثابت صحیح pos = صف * 4 + (lk & 3);
                lo_pl[(g >> 2) * 128 + pos] |= (uint8_t) (q2_3_K_get_low2(b->qs, kt_local * 32 + lk) << ((g & 3) * 2));
            }
#اگر_ختم
            برائے (صحیح sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + صف] = b->scales[kt_local * 2 + sub];
            }
            اگر (صف / 8 == kt_local % 4) {
                dd_pl[صف % 8] = kt_local < 4 ? b->d : (ggml_half) (b->dmin ^ 0x8000);
            }
        });
}

ساکن خالی repack_tiled_q2_K(خالی * ڈیٹا, ثابت ggml_tensor * t, size_t انحراف, size_t size) {
    unpack_kquant_from_tiled_grid<بلاک_q2_K>(ڈیٹا, t, انحراف, size, HTP_TENSOR_TILE_سائز_Q2_K,
        [](بلاک_q2_K * b, ثابت uint8_t * sb_tiles, size_t tile_سائز, صحیح صف) {
            برائے (صحیح kt_local = 0; kt_local < 8; kt_local++) {
                ثابت uint8_t *   tile  = sb_tiles + kt_local * tile_سائز;
                ثابت uint8_t *   lo_pl = tile;
                ثابت uint8_t *   sc_pl = tile + 256;
                ثابت ggml_half * dd_pl = (ثابت ggml_half *) (tile + 320);

                برائے (صحیح lk = 0; lk < 32; lk++) {
                    ثابت صحیح e   = kt_local * 32 + lk;
                    ثابت صحیح g   = lk >> 2;
                    ثابت صحیح pos = صف * 4 + (lk & 3);
                    ثابت uint8_t lo = (lo_pl[(g >> 2) * 128 + pos] >> ((g & 3) * 2)) & 3;
                    b->qs[(e / 128) * 32 + e % 32] |= (uint8_t) (lo << (2 * ((e % 128) / 32)));
                }

                برائے (صحیح sub = 0; sub < 2; sub++) {
                    b->scales[kt_local * 2 + sub] = sc_pl[sub * 32 + صف];
                }
                اگر (صف / 8 == kt_local % 4) {
                    اگر (kt_local < 4) {
                        b->d = dd_pl[صف % 8];
                    } ورنہ {
                        b->dmin = (ggml_half) (dd_pl[صف % 8] ^ 0x8000);
                    }
                }
            }
        });
}

ساکن اندرونی خالی get_scale_min_k4(صحیح j, ثابت uint8_t * q, uint8_t * d, uint8_t * m) {
    اگر (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } ورنہ {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

ساکن اندرونی خالی pack_k4_scales_mins(uint8_t * scales, ggml_half * d, ggml_half * dmin,
                                       ثابت اعشاری * sub_scales, ثابت اعشاری * sub_mins) {
    اعشاری max_scale = 0.0f;
    اعشاری max_min   = 0.0f;
    برائے (صحیح j = 0; j < 8; j++) {
        اگر (sub_scales[j] > max_scale) max_scale = sub_scales[j];
        اگر (sub_mins[j]   > max_min)   max_min   = sub_mins[j];
    }

    اعشاری inv_scale = 0.0f;
    اگر (max_scale > 0.0f) {
        *d = GGML_FP32_مقصد_FP16(max_scale / 63.0f);
        ثابت اعشاری d_actual = GGML_FP16_مقصد_FP32(*d);
        inv_scale = (d_actual > 0.0f) ? (1.0f / d_actual) : 0.0f;
    } ورنہ {
        *d = GGML_FP32_مقصد_FP16(0.0f);
    }

    اعشاری inv_min = 0.0f;
    اگر (max_min > 0.0f) {
        *dmin = GGML_FP32_مقصد_FP16(max_min / 63.0f);
        ثابت اعشاری dmin_actual = GGML_FP16_مقصد_FP32(*dmin);
        inv_min = (dmin_actual > 0.0f) ? (1.0f / dmin_actual) : 0.0f;
    } ورنہ {
        *dmin = GGML_FP32_مقصد_FP16(0.0f);
    }

    برائے (صحیح j = 0; j < 8; j++) {
        uint8_t ls = (std::min)((uint8_t) 63, (uint8_t) roundf(inv_scale * sub_scales[j]));
        uint8_t lm = (std::min)((uint8_t) 63, (uint8_t) roundf(inv_min * sub_mins[j]));
        اگر (j < 4) {
            scales[j]     = ls;
            scales[j + 4] = lm;
        } ورنہ {
            scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            scales[j - 4] |= ((ls >> 4) << 6);
            scales[j - 0] |= ((lm >> 4) << 6);
        }
    }
}

سانچہ <قسم_نام بلاک_T>
ساکن اندرونی خالی repack_k4_quants(uint8_t * tile_dst, ثابت بلاک_T * b, صحیح صف, صحیح kt_local,
                                    int64_t sb_فی_صف, منطقی use_neon) {
#اگر defined(__ARM_NEON) || defined(__aarch64__)
    اگر (use_neon) {
        اگر ((صف & 7) == 0) {
            ثابت uint8_t * qs[8];
            برائے (صحیح rr = 0; rr < 8; ++rr) {
                qs[rr] = b[rr * sb_فی_صف].qs + (kt_local / 2) * 32;
            }
            repack_neon_k4_صفیں(qs, tile_dst, صف, (kt_local & 1) != 0);
        }
        واپس;
    }
#ورنہ
    GGML_غیر_مستعمل(sb_فی_صف);
    GGML_غیر_مستعمل(use_neon);
#اگر_ختم
    ثابت uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
    ثابت صحیح سرکاؤ = (kt_local & 1) ? 4 : 0;

    برائے (صحیح cp = 0; cp < 16; cp++) {
        ثابت uint8_t q0 = (qs_sub[2 * cp + 0] >> سرکاؤ) & 0x0F;
        ثابت uint8_t q1 = (qs_sub[2 * cp + 1] >> سرکاؤ) & 0x0F;
        tile_dst[cp * 32 + صف] = (uint8_t) ((q1 << 4) | q0);
    }
}

ساکن خالی repack_q4_K_tiled(ggml_tensor * t, ثابت خالی * ڈیٹا, size_t انحراف, size_t size) {
    ثابت int64_t sb_فی_صف = t->ne[0] / QK_K;
    ثابت منطقی    use_neon   = (t->ne[1] % 8 == 0);
    repack_kquant_مقصد_tiled_grid<بلاک_q4_K>(t, ڈیٹا, انحراف, size, HTP_TENSOR_TILE_سائز_Q4_K,
        [sb_فی_صف, use_neon](uint8_t * tile_dst, ثابت بلاک_q4_K * b, صحیح صف, صحیح kt_local) {
            ثابت اعشاری d    = GGML_FP16_مقصد_FP32(b->d);
            ثابت اعشاری dmin = GGML_FP16_مقصد_FP32(b->dmin);

            uint8_t sc, m;
            get_scale_min_k4(kt_local, b->scales, &sc, &m);

            repack_k4_quants(tile_dst, b, صف, kt_local, sb_فی_صف, use_neon);

            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
            scale_dst[2 * صف + 0] = GGML_FP32_مقصد_FP16(d * (اعشاری) sc);
            scale_dst[2 * صف + 1] = GGML_FP32_مقصد_FP16(-dmin * (اعشاری) m);
        });
}

ساکن خالی repack_tiled_q4_K(خالی * ڈیٹا, ثابت ggml_tensor * t, size_t انحراف, size_t size) {
    unpack_kquant_from_tiled_grid<بلاک_q4_K>(ڈیٹا, t, انحراف, size, HTP_TENSOR_TILE_سائز_Q4_K,
        [](بلاک_q4_K * b, ثابت uint8_t * sb_tiles, size_t tile_سائز, صحیح صف) {
            اعشاری sub_scales[8];
            اعشاری sub_mins[8];

            برائے (صحیح kt_local = 0; kt_local < 8; kt_local++) {
                ثابت uint8_t *   tile_src  = sb_tiles + kt_local * tile_سائز;
                ثابت ggml_half * scale_src = (ثابت ggml_half *) (tile_src + 512);

                uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
                ثابت صحیح سرکاؤ  = (kt_local & 1) ? 4 : 0;

                برائے (صحیح cp = 0; cp < 16; cp++) {
                    ثابت uint8_t val = tile_src[cp * 32 + صف];
                    qs_sub[2 * cp + 0] |= (uint8_t) ((val & 0x0F) << سرکاؤ);
                    qs_sub[2 * cp + 1] |= (uint8_t) ((val >> 4)   << سرکاؤ);
                }

                ثابت اعشاری D = GGML_FP16_مقصد_FP32(scale_src[2 * صف + 0]);
                ثابت اعشاری M = GGML_FP16_مقصد_FP32(scale_src[2 * صف + 1]);
                sub_scales[kt_local] = (D > 0.0f) ? D : 0.0f;
                sub_mins[kt_local]   = (-M > 0.0f) ? -M : 0.0f;
            }

            pack_k4_scales_mins(b->scales, &b->d, &b->dmin, sub_scales, sub_mins);
        });
}

ساکن خالی repack_q5_K_tiled(ggml_tensor * t, ثابت خالی * ڈیٹا, size_t انحراف, size_t size) {
    ثابت int64_t sb_فی_صف = t->ne[0] / QK_K;
    ثابت منطقی    use_neon   = (t->ne[1] % 8 == 0);
    repack_kquant_مقصد_tiled_grid<بلاک_q5_K>(t, ڈیٹا, انحراف, size, HTP_TENSOR_TILE_سائز_Q5_K,
        [sb_فی_صف, use_neon](uint8_t * tile_dst, ثابت بلاک_q5_K * b, صحیح صف, صحیح kt_local) {
            ثابت اعشاری d     = GGML_FP16_مقصد_FP32(b->d);
            ثابت اعشاری dmin  = GGML_FP16_مقصد_FP32(b->dmin);
            uint8_t * plane   = tile_dst + 640;

            uint8_t sc, m;
            get_scale_min_k4(kt_local, b->scales, &sc, &m);

            repack_k4_quants(tile_dst, b, صف, kt_local, sb_فی_صف, use_neon);

            ثابت uint8_t hbit = (uint8_t) (1 << kt_local);
#اگر defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q5_k_qh(b->qh, hbit, plane, صف);
#ورنہ
            برائے (صحیح cp = 0; cp < 16; cp++) {
                ثابت صحیح i    = cp / 4;
                ثابت صحیح lane = (cp % 4) * 32 + صف;
                اگر (b->qh[2 * cp + 0] & hbit) {
                    plane[lane] |= (uint8_t) (1 << (2 * i));
                }
                اگر (b->qh[2 * cp + 1] & hbit) {
                    plane[lane] |= (uint8_t) (1 << (2 * i + 1));
                }
            }
#اگر_ختم

            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
            scale_dst[2 * صف + 0] = GGML_FP32_مقصد_FP16(d * (اعشاری) sc);
            scale_dst[2 * صف + 1] = GGML_FP32_مقصد_FP16(-dmin * (اعشاری) m);
        });
}

ساکن خالی repack_tiled_q5_K(خالی * ڈیٹا, ثابت ggml_tensor * t, size_t انحراف, size_t size) {
    unpack_kquant_from_tiled_grid<بلاک_q5_K>(ڈیٹا, t, انحراف, size, HTP_TENSOR_TILE_سائز_Q5_K,
        [](بلاک_q5_K * b, ثابت uint8_t * sb_tiles, size_t tile_سائز, صحیح صف) {
            اعشاری sub_scales[8];
            اعشاری sub_mins[8];

            برائے (صحیح kt_local = 0; kt_local < 8; kt_local++) {
                ثابت uint8_t *   tile_src  = sb_tiles + kt_local * tile_سائز;
                ثابت uint8_t *   plane     = tile_src + 640;
                ثابت ggml_half * scale_src = (ثابت ggml_half *) (tile_src + 512);

                uint8_t * qs_sub   = b->qs + (kt_local / 2) * 32;
                ثابت صحیح سرکاؤ    = (kt_local & 1) ? 4 : 0;
                ثابت uint8_t hbit = (uint8_t) (1 << kt_local);

                برائے (صحیح cp = 0; cp < 16; cp++) {
                    ثابت uint8_t val = tile_src[cp * 32 + صف];
                    qs_sub[2 * cp + 0] |= (uint8_t) ((val & 0x0F) << سرکاؤ);
                    qs_sub[2 * cp + 1] |= (uint8_t) ((val >> 4)   << سرکاؤ);

                    ثابت صحیح i    = cp / 4;
                    ثابت صحیح lane = (cp % 4) * 32 + صف;
                    اگر (plane[lane] & (1 << (2 * i))) {
                        b->qh[2 * cp + 0] |= hbit;
                    }
                    اگر (plane[lane] & (1 << (2 * i + 1))) {
                        b->qh[2 * cp + 1] |= hbit;
                    }
                }

                ثابت اعشاری D = GGML_FP16_مقصد_FP32(scale_src[2 * صف + 0]);
                ثابت اعشاری M = GGML_FP16_مقصد_FP32(scale_src[2 * صف + 1]);
                sub_scales[kt_local] = (D > 0.0f) ? D : 0.0f;
                sub_mins[kt_local]   = (-M > 0.0f) ? -M : 0.0f;
            }

            pack_k4_scales_mins(b->scales, &b->d, &b->dmin, sub_scales, sub_mins);
        });
}

ساکن اندرونی خالی q1_0_tiled_ثنائی_ہندسہ_location(صحیح k, صحیح صف, صحیح & byte_انحراف, صحیح & ثنائی_ہندسہ) {
    ثابت صحیح quarter = k / 32;
    ثابت صحیح kq      = k % 32;

    byte_انحراف = quarter * 128 + 4 * صف + (kq % 4);
    ثنائی_ہندسہ         = kq / 4;
}

ساکن خالی repack_q1_0_tiled(ggml_tensor * t, ثابت خالی * ڈیٹا, size_t انحراف, size_t size) {
    ثابت بلاک_q1_0 * src_میٹرکس = (ثابت بلاک_q1_0 *) ڈیٹا;
    ثابت int64_t ne0 = t->ne[0];
    ثابت int64_t ne1 = t->ne[1];
    ثابت int64_t ne2 = t->ne[2];
    ثابت int64_t ne3 = t->ne[3];
    ثابت صحیح n_ستون_tiles = hex_باری_اوپر(ne1, 32) / 32;
    ثابت صحیح n_k_tiles = ne0 / QK1_0;
    ثابت size_t tile_سائز = HTP_TENSOR_TILE_سائز_Q1_0;
    ثابت size_t میٹرکس_سائز = (size_t) n_ستون_tiles * n_k_tiles * tile_سائز;
    ثابت size_t قاش_سائز = ne1 * ggml_صف_سائز(t->type, ne0);
    ثابت int64_t آغاز_قاش = انحراف / قاش_سائز;
    ثابت int64_t انتہا_قاش = (std::min)((int64_t) ((انحراف + size + قاش_سائز - 1) / قاش_سائز), ne2 * ne3);

    GGML_ASSERT(ne0 % QK1_0 == 0);

    برائے (int64_t قاش_idx = آغاز_قاش; قاش_idx < انتہا_قاش; قاش_idx++) {
        ثابت بلاک_q1_0 * src_قاش = src_میٹرکس + (قاش_idx - آغاز_قاش) * ne1 * n_k_tiles;
        uint8_t * میٹرکس_dst = (uint8_t *) t->ڈیٹا + قاش_idx * میٹرکس_سائز;

        repack_parallel_for_ct(میٹرکس_سائز, n_ستون_tiles, [&](صحیح ct0, صحیح ct1) {
            برائے (صحیح ct = ct0; ct < ct1; ct++) {
                برائے (صحیح kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = میٹرکس_dst + ((size_t) ct * n_k_tiles + kt) * tile_سائز;
                    ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
                    memset(tile_dst, 0, tile_سائز);

                    برائے (صحیح صف = 0; صف < 32; صف++) {
                        ثابت int64_t r = ct * 32 + صف;
                        اگر (r >= ne1) {
                            جاری;
                        }

                        ثابت بلاک_q1_0 & بلاک = src_قاش[r * n_k_tiles + kt];
                        scale_dst[صف] = بلاک.d;
#اگر defined(__ARM_NEON) || defined(__aarch64__)
                        repack_neon_q1_0_بلاک(بلاک.qs, tile_dst, صف);
#ورنہ
                        برائے (صحیح k = 0; k < QK1_0; k++) {
                            صحیح byte_انحراف;
                            صحیح ثنائی_ہندسہ;
                            q1_0_tiled_ثنائی_ہندسہ_location(k, صف, byte_انحراف, ثنائی_ہندسہ);
                            اگر (بلاک.qs[k / 8] & (1u << (k % 8))) {
                                tile_dst[byte_انحراف] |= (uint8_t) (1u << ثنائی_ہندسہ);
                            }
                        }
#اگر_ختم
                    }
                }
            }
        });
    }
}

ساکن خالی repack_tiled_q1_0(خالی * ڈیٹا, ثابت ggml_tensor * t, size_t انحراف, size_t size) {
    بلاک_q1_0 * dst_میٹرکس = (بلاک_q1_0 *) ڈیٹا;
    ثابت int64_t ne0 = t->ne[0];
    ثابت int64_t ne1 = t->ne[1];
    ثابت int64_t ne2 = t->ne[2];
    ثابت int64_t ne3 = t->ne[3];
    ثابت صحیح n_ستون_tiles = hex_باری_اوپر(ne1, 32) / 32;
    ثابت صحیح n_k_tiles = ne0 / QK1_0;
    ثابت size_t tile_سائز = HTP_TENSOR_TILE_سائز_Q1_0;
    ثابت size_t میٹرکس_سائز = (size_t) n_ستون_tiles * n_k_tiles * tile_سائز;
    ثابت size_t صف_سائز_بائٹیں = ggml_صف_سائز(t->type, ne0);
    ثابت size_t قاش_سائز = ne1 * صف_سائز_بائٹیں;
    ثابت int64_t آغاز_قاش = انحراف / قاش_سائز;
    ثابت int64_t انتہا_قاش = (std::min)((int64_t) ((انحراف + size + قاش_سائز - 1) / قاش_سائز), ne2 * ne3);

    GGML_ASSERT(ne0 % QK1_0 == 0);

    برائے (int64_t قاش_idx = آغاز_قاش; قاش_idx < انتہا_قاش; قاش_idx++) {
        ثابت size_t cur_آغاز_byte = (std::max)(انحراف, (size_t) قاش_idx * قاش_سائز);
        ثابت size_t cur_انتہا_byte = (std::min)(انحراف + size, (size_t) (قاش_idx + 1) * قاش_سائز);
        ثابت int64_t آغاز_صف = (cur_آغاز_byte - (size_t) قاش_idx * قاش_سائز) / صف_سائز_بائٹیں;
        ثابت int64_t انتہا_صف = (std::min)((int64_t) ((cur_انتہا_byte - (size_t) قاش_idx * قاش_سائز + صف_سائز_بائٹیں - 1) / صف_سائز_بائٹیں), ne1);
        بلاک_q1_0 * dst_قاش = dst_میٹرکس + (cur_آغاز_byte - انحراف) / حجم(بلاک_q1_0);
        ثابت uint8_t * میٹرکس_src = (ثابت uint8_t *) t->ڈیٹا + قاش_idx * میٹرکس_سائز;

        برائے (صحیح ct = آغاز_صف / 32; ct < (انتہا_صف + 31) / 32; ct++) {
            برائے (صحیح kt = 0; kt < n_k_tiles; kt++) {
                ثابت uint8_t * tile_src = میٹرکس_src + ((size_t) ct * n_k_tiles + kt) * tile_سائز;
                ثابت ggml_half * scale_src = (ثابت ggml_half *) (tile_src + 512);

                برائے (صحیح صف = 0; صف < 32; صف++) {
                    ثابت int64_t r = ct * 32 + صف;
                    اگر (r < آغاز_صف || r >= انتہا_صف) {
                        جاری;
                    }

                    بلاک_q1_0 & بلاک = dst_قاش[(r - آغاز_صف) * n_k_tiles + kt];
                    memset(بلاک.qs, 0, حجم(بلاک.qs));
                    بلاک.d = scale_src[صف];
                    برائے (صحیح k = 0; k < QK1_0; k++) {
                        صحیح byte_انحراف;
                        صحیح ثنائی_ہندسہ;
                        q1_0_tiled_ثنائی_ہندسہ_location(k, صف, byte_انحراف, ثنائی_ہندسہ);
                        اگر (tile_src[byte_انحراف] & (1u << ثنائی_ہندسہ)) {
                            بلاک.qs[k / 8] |= (uint8_t) (1u << (k % 8));
                        }
                    }
                }
            }
        }
    }
}

// --- Top-level Repack / Unpack Dispatchers ---

خالی htp_repack_tensor_tiled(ggml_tensor * tensor, ثابت خالی * ڈیٹا, size_t size) {
    انتخاب (tensor->type) {
        حالت GGML_TYPE_Q1_0:
            repack_q1_0_tiled(tensor, ڈیٹا, 0, size);
            توڑو;
        حالت GGML_TYPE_Q4_0:
        حالت GGML_TYPE_IQ4_NL:
            repack_q4_0_tiled(tensor, ڈیٹا, 0, size);
            توڑو;
        حالت GGML_TYPE_Q4_1:
            repack_q4_1_tiled(tensor, ڈیٹا, 0, size);
            توڑو;
        حالت GGML_TYPE_Q8_0:
            repack_q8_0_tiled(tensor, ڈیٹا, 0, size);
            توڑو;
        حالت GGML_TYPE_MXFP4:
            repack_mxfp4_tiled(tensor, ڈیٹا, 0, size);
            توڑو;
        حالت GGML_TYPE_Q2_0:
            repack_q2_0_tiled(tensor, ڈیٹا, 0, size);
            توڑو;
        حالت GGML_TYPE_Q2_K:
            repack_q2_K_tiled(tensor, ڈیٹا, 0, size);
            توڑو;
        حالت GGML_TYPE_Q3_K:
            repack_q3_K_tiled(tensor, ڈیٹا, 0, size);
            توڑو;
        حالت GGML_TYPE_Q4_K:
            repack_q4_K_tiled(tensor, ڈیٹا, 0, size);
            توڑو;
        حالت GGML_TYPE_Q5_K:
            repack_q5_K_tiled(tensor, ڈیٹا, 0, size);
            توڑو;
        حالت GGML_TYPE_Q6_K:
            repack_q6_K_tiled(tensor, ڈیٹا, 0, size);
            توڑو;
        حالت GGML_TYPE_F16:
            repack_float_مقصد_f16_tiled<uint16_t>(tensor, ڈیٹا, 0, size,
                [](uint16_t v) { واپس v; });
            توڑو;
        حالت GGML_TYPE_F32:
            repack_float_مقصد_f16_tiled<اعشاری>(tensor, ڈیٹا, 0, size,
                [](اعشاری v) { واپس (uint16_t) GGML_FP32_مقصد_FP16(v); });
            توڑو;
        حالت GGML_TYPE_BF16:
            repack_float_مقصد_f16_tiled<ggml_bf16_t>(tensor, ڈیٹا, 0, size,
                [](ggml_bf16_t v) { واپس (uint16_t) GGML_FP32_مقصد_FP16(GGML_BF16_مقصد_FP32(v)); });
            توڑو;
        طےشدہ:
            توڑو;
    }
}

خالی htp_unpack_tensor_tiled(خالی * ڈیٹا, ثابت ggml_tensor * tensor, size_t انحراف, size_t size) {
    GGML_ASSERT(انحراف + size <= ggml_nbytes(tensor));

    انتخاب (tensor->type) {
        حالت GGML_TYPE_Q1_0:
            repack_tiled_q1_0(ڈیٹا, tensor, انحراف, size);
            توڑو;
        حالت GGML_TYPE_Q4_0:
        حالت GGML_TYPE_IQ4_NL:
            repack_tiled_q4_0(ڈیٹا, tensor, انحراف, size);
            توڑو;
        حالت GGML_TYPE_Q4_1:
            repack_tiled_q4_1(ڈیٹا, tensor, انحراف, size);
            توڑو;
        حالت GGML_TYPE_Q8_0:
            repack_tiled_q8_0(ڈیٹا, tensor, انحراف, size);
            توڑو;
        حالت GGML_TYPE_MXFP4:
            repack_tiled_mxfp4(ڈیٹا, tensor, انحراف, size);
            توڑو;
        حالت GGML_TYPE_Q2_0:
            repack_tiled_q2_0(ڈیٹا, tensor, انحراف, size);
            توڑو;
        حالت GGML_TYPE_Q2_K:
            repack_tiled_q2_K(ڈیٹا, tensor, انحراف, size);
            توڑو;
        حالت GGML_TYPE_Q3_K:
            repack_tiled_q3_K(ڈیٹا, tensor, انحراف, size);
            توڑو;
        حالت GGML_TYPE_Q4_K:
            repack_tiled_q4_K(ڈیٹا, tensor, انحراف, size);
            توڑو;
        حالت GGML_TYPE_Q5_K:
            repack_tiled_q5_K(ڈیٹا, tensor, انحراف, size);
            توڑو;
        حالت GGML_TYPE_Q6_K:
            repack_tiled_q6_K(ڈیٹا, tensor, انحراف, size);
            توڑو;
        حالت GGML_TYPE_F16:
            unpack_f16_tiled_مقصد_float<uint16_t>(ڈیٹا, tensor, انحراف, size,
                [](uint16_t v) { واپس v; });
            توڑو;
        حالت GGML_TYPE_F32:
            unpack_f16_tiled_مقصد_float<اعشاری>(ڈیٹا, tensor, انحراف, size,
                [](uint16_t v) { واپس ggml_fp16_مقصد_fp32((ggml_fp16_t) v); });
            توڑو;
        حالت GGML_TYPE_BF16:
            unpack_f16_tiled_مقصد_float<ggml_bf16_t>(ڈیٹا, tensor, انحراف, size,
                [](uint16_t v) { واپس ggml_fp32_مقصد_bf16(ggml_fp16_مقصد_fp32((ggml_fp16_t) v)); });
            توڑو;
        طےشدہ:
            حافظہ_نقل(ڈیٹا, (ثابت حرف *) tensor->ڈیٹا + انحراف, size);
            توڑو;
    }
}
