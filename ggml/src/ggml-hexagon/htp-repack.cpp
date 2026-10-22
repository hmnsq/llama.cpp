#ضمن "htp-repack.h"

#تعریف_کن GGML_COMMON_IMPL_CPP
#ضمن "ggml-common.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-quants.h"

#ضمن <algorithm>
#ضمن <cmath>
#ضمن <cstring>
#ضمن <initializer_list>
#ضمن <thread>

#إذا defined(__ARM_NEON) || defined(__aarch64__)
#    include <arm_neon.h>
#پایان_اگر

// --- 4-bit block-32 nibble helpers (shared by Q4_0, IQ4_NL, Q4_1, MXFP4) ---

ساكن مضمن فراغ unpack_nibbles_32(uint8_t * dst, ثابت uint8_t * qs) {
    لكل (بیعلامت صحيح i = 0; i < 16; ++i) {
        dst[i + 0]  = qs[i] & 0x0F;
        dst[i + 16] = qs[i] >> 4;
    }
}

ساكن مضمن فراغ pack_nibbles_32(uint8_t * qs, ثابت uint8_t * src) {
    لكل (بیعلامت صحيح i = 0; i < 16; ++i) {
        qs[i] = src[i + 0] | (src[i + 16] << 4);
    }
}

ساكن مضمن فراغ transpose_nibbles_مقصد_tile(uint8_t * tile_dst, ثابت uint8_t tile_quants[32][32]) {
    لكل (صحيح cp = 0; cp < 16; cp++) {
        لكل (صحيح ردیف = 0; ردیف < 32; ردیف++) {
            tile_dst[cp * 32 + ردیف] = (tile_quants[ردیف][2 * cp + 1] << 4) | tile_quants[ردیف][2 * cp];
        }
    }
}

ساكن مضمن فراغ transpose_tile_مقصد_nibbles(uint8_t tile_quants[32][32], ثابت uint8_t * tile_src) {
    لكل (صحيح cp = 0; cp < 16; cp++) {
        لكل (صحيح ردیف = 0; ردیف < 32; ردیف++) {
            ثابت uint8_t val = tile_src[cp * 32 + ردیف];
            tile_quants[ردیف][2 * cp + 0] = val & 0x0F;
            tile_quants[ردیف][2 * cp + 1] = val >> 4;
        }
    }
}

#إذا defined(__ARM_NEON) || defined(__aarch64__)
ساكن فراغ repack_neon_transpose8x8(ثابت uint8x8_t ردیفها[8], uint8x8_t columns[8]) {
    ثابت uint8x8x2_t t0 = vtrn_u8(ردیفها[0], ردیفها[1]);
    ثابت uint8x8x2_t t1 = vtrn_u8(ردیفها[2], ردیفها[3]);
    ثابت uint8x8x2_t t2 = vtrn_u8(ردیفها[4], ردیفها[5]);
    ثابت uint8x8x2_t t3 = vtrn_u8(ردیفها[6], ردیفها[7]);
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

ساكن uint8x8_t repack_neon_pack_4bit_جفتها(uint8x8_t اولی, uint8x8_t second, منطقي بالا) {
    ثابت uint8x8x2_t separated = vuzp_u8(اولی, second);
    ثابت uint8x8_t ماسک = vdup_n_u8(0x0F);
    uint8x8_t even = separated.val[0];
    uint8x8_t odd  = separated.val[1];

    إذا (بالا) {
        even = vshr_n_u8(even, 4);
        odd  = vshr_n_u8(odd, 4);
    }

    even = vand_u8(even, ماسک);
    odd  = vand_u8(odd, ماسک);
    أرجع vorr_u8(vshl_n_u8(odd, 4), even);
}

ساكن فراغ repack_neon_4bit_ردیفها(ثابت uint8_t * qs[8], uint8_t * tile_dst, صحيح row0) {
    uint8x8_t کم_ردیفها[8];
    uint8x8_t بالا_ردیفها[8];
    uint8x8_t columns[8];

    لكل (صحيح rr = 0; rr < 8; ++rr) {
        ثابت uint8x16_t packed = vld1q_u8(qs[rr]);
        ثابت uint8x8_t اولی = vget_کم_u8(packed);
        ثابت uint8x8_t second = vget_بالا_u8(packed);
        کم_ردیفها[rr]  = repack_neon_pack_4bit_جفتها(اولی, second, خطأ);
        بالا_ردیفها[rr] = repack_neon_pack_4bit_جفتها(اولی, second, درست);
    }

    repack_neon_transpose8x8(کم_ردیفها, columns);
    لكل (صحيح cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + cp * 32 + row0, columns[cp]);
    }

    repack_neon_transpose8x8(بالا_ردیفها, columns);
    لكل (صحيح cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + (cp + 8) * 32 + row0, columns[cp]);
    }
}

ساكن فراغ repack_neon_k4_ردیفها(ثابت uint8_t * qs[8], uint8_t * tile_dst, صحيح row0, منطقي بالا) {
    uint8x8_t اولی_ردیفها[8];
    uint8x8_t دومی_ردیفها[8];
    uint8x8_t columns[8];

    لكل (صحيح rr = 0; rr < 8; ++rr) {
        اولی_ردیفها[rr] = repack_neon_pack_4bit_جفتها(vld1_u8(qs[rr]), vld1_u8(qs[rr] + 8), بالا);
        دومی_ردیفها[rr] = repack_neon_pack_4bit_جفتها(vld1_u8(qs[rr] + 16), vld1_u8(qs[rr] + 24), بالا);
    }

    repack_neon_transpose8x8(اولی_ردیفها, columns);
    لكل (صحيح cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + cp * 32 + row0, columns[cp]);
    }

    repack_neon_transpose8x8(دومی_ردیفها, columns);
    لكل (صحيح cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + (cp + 8) * 32 + row0, columns[cp]);
    }
}

ساكن فراغ repack_neon_q8_ردیفها(ثابت uint8_t * qs[8], uint8_t * tile_dst, صحيح row0) {
    uint8x8_t ردیفها[8];
    uint8x8_t columns[8];

    لكل (صحيح cp = 0; cp < 16; cp += 4) {
        لكل (صحيح rr = 0; rr < 8; ++rr) {
            ردیفها[rr] = vld1_u8(qs[rr] + 2 * cp);
        }
        repack_neon_transpose8x8(ردیفها, columns);

        لكل (صحيح pair = 0; pair < 4; ++pair) {
            ثابت uint8x8x2_t interleaved = vzip_u8(columns[2 * pair], columns[2 * pair + 1]);
            uint8_t * dst = tile_dst + (cp + pair) * 64 + 2 * row0;
            vst1_u8(dst, interleaved.val[0]);
            vst1_u8(dst + 8, interleaved.val[1]);
        }
    }
}

ساكن مضمن فراغ repack_store_u32(uint8_t * dst, uint32_t v) {
    کپی_حافظه(dst, &v, حجم(v));
}

ساكن مضمن uint32x4_t repack_neon_transpose_2bit_4x4(uint32x4_t w) {
    uint32x4_t t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 6)), vdupq_n_u32(0x00CC00CCu));
    w = veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 6)));
    t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 12)), vdupq_n_u32(0x0000F0F0u));
    أرجع veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 12)));
}

ساكن مضمن فراغ repack_neon_q2_0_بلوک(ثابت uint8_t * qs, uint8_t * tile, صحيح ردیف) {
    ثابت uint32x4_t w = repack_neon_transpose_2bit_4x4(vreinterpretq_u32_u8(vld1q_u8(qs)));
    repack_store_u32(tile + 0 * 128 + 4 * ردیف, vgetq_lane_u32(w, 0));
    repack_store_u32(tile + 1 * 128 + 4 * ردیف, vgetq_lane_u32(w, 1));
    repack_store_u32(tile + 2 * 128 + 4 * ردیف, vgetq_lane_u32(w, 2));
    repack_store_u32(tile + 3 * 128 + 4 * ردیف, vgetq_lane_u32(w, 3));
}

ساكن مضمن فراغ repack_neon_q1_0_بلوک(ثابت uint8_t * qs, uint8_t * tile, صحيح ردیف) {
    uint32x4_t w = vreinterpretq_u32_u8(vld1q_u8(qs));
    uint32x4_t t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 1)), vdupq_n_u32(0x22222222u));
    w = veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 1)));
    t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 3)), vdupq_n_u32(0x0A0A0A0Au));
    w = veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 3)));
    w = repack_neon_transpose_2bit_4x4(w);
    repack_store_u32(tile + 0 * 128 + 4 * ردیف, vgetq_lane_u32(w, 0));
    repack_store_u32(tile + 1 * 128 + 4 * ردیف, vgetq_lane_u32(w, 1));
    repack_store_u32(tile + 2 * 128 + 4 * ردیف, vgetq_lane_u32(w, 2));
    repack_store_u32(tile + 3 * 128 + 4 * ردیف, vgetq_lane_u32(w, 3));
}

ساكن مضمن فراغ repack_neon_pack_2bit_plane(uint8x16_t v0, uint8x16_t v1, uint8_t * dst, صحيح ردیف) {
    ثابت uint32x4x2_t z   = vuzpq_u32(vreinterpretq_u32_u8(v0), vreinterpretq_u32_u8(v1));
    ثابت uint32x4_t   p01 = vorrq_u32(z.val[0], vshlq_n_u32(z.val[1], 2));
    ثابت uint32x2x2_t u   = vuzp_u32(vget_کم_u32(p01), vget_بالا_u32(p01));
    ثابت uint32x2_t   بیرون = vorr_u32(u.val[0], vshl_n_u32(u.val[1], 4));
    repack_store_u32(dst + 0 * 128 + ردیف * 4, vget_lane_u32(بیرون, 0));
    repack_store_u32(dst + 1 * 128 + ردیف * 4, vget_lane_u32(بیرون, 1));
}

ساكن مضمن فراغ repack_neon_q2_3_k_low2(ثابت uint8_t * qs, صحيح kt_local, uint8_t * lo_pl, صحيح ردیف) {
    ثابت uint8_t *  qs_sub = qs + (kt_local / 4) * 32;
    ثابت int8x16_t  لغزش  = vdupq_n_s8(-2 * (kt_local % 4));
    ثابت uint8x16_t ماسک   = vdupq_n_u8(0x03);
    ثابت uint8x16_t v0     = vandq_u8(vshlq_u8(vld1q_u8(qs_sub +  0), لغزش), ماسک);
    ثابت uint8x16_t v1     = vandq_u8(vshlq_u8(vld1q_u8(qs_sub + 16), لغزش), ماسک);
    repack_neon_pack_2bit_plane(v0, v1, lo_pl, ردیف);
}

ساكن مضمن فراغ repack_neon_q3_k_hmask(ثابت uint8_t * hmask, صحيح kt_local, uint8_t * neg_pl, صحيح ردیف) {
    ساكن ثابت uint32_t ماسک_واژگان[4] = { 0x01010101u, 0x02020202u, 0x04040404u, 0x08080808u };
    ثابت uint8x16_t m0   = vreinterpretq_u8_u32(vld1q_u32(ماسک_واژگان));
    ثابت uint8x16_t m1   = vshlq_n_u8(m0, 4);
    ثابت uint8x16_t hbit = vdupq_n_u8((uint8_t) (1u << kt_local));
    ثابت uint8x16_t b0   = vbicq_u8(m0, vtstq_u8(vld1q_u8(hmask +  0), hbit));
    ثابت uint8x16_t b1   = vbicq_u8(m1, vtstq_u8(vld1q_u8(hmask + 16), hbit));
    ثابت uint32x4_t b01  = vreinterpretq_u32_u8(vorrq_u8(b0, b1));
    ثابت uint32x2_t f0   = vorr_u32(vget_کم_u32(b01), vget_بالا_u32(b01));
    ثابت uint32x2_t f1   = vpadd_u32(f0, f0);
    repack_store_u32(neg_pl + ردیف * 4, vget_lane_u32(f1, 0));
}

ساكن مضمن فراغ repack_neon_q6_k(ثابت بلوک_q6_K * b, صحيح kt_local, uint8_t * lo_pl, uint8_t * hi_pl, صحيح ردیف) {
    ثابت صحيح        c      = kt_local / 4;
    ثابت صحيح        g_sb   = kt_local % 4;
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
    repack_store_u32(lo_pl + 0 * 128 + ردیف * 4, vgetq_lane_u32(lout, 0));
    repack_store_u32(lo_pl + 1 * 128 + ردیف * 4, vgetq_lane_u32(lout, 1));
    repack_store_u32(lo_pl + 2 * 128 + ردیف * 4, vgetq_lane_u32(lout, 2));
    repack_store_u32(lo_pl + 3 * 128 + ردیف * 4, vgetq_lane_u32(lout, 3));

    ثابت uint8x16_t hi0 = vandq_u8(vshlq_u8(vld1q_u8(qh +  0), hshift), mask2);
    ثابت uint8x16_t hi1 = vandq_u8(vshlq_u8(vld1q_u8(qh + 16), hshift), mask2);
    repack_neon_pack_2bit_plane(hi0, hi1, hi_pl, ردیف);
}

ساكن مضمن فراغ repack_neon_q5_k_qh(ثابت uint8_t * qh, uint8_t hbit, uint8_t * plane, صحيح ردیف) {
    ساكن ثابت uint8_t ماسک_بایتها[16] = { 1, 2, 1, 2, 1, 2, 1, 2, 4, 8, 4, 8, 4, 8, 4, 8 };
    ثابت uint8x16_t m0   = vld1q_u8(ماسک_بایتها);
    ثابت uint8x16_t m1   = vshlq_n_u8(m0, 4);
    ثابت uint8x16_t vbit = vdupq_n_u8(hbit);
    ثابت uint8x16_t b0   = vandq_u8(vtstq_u8(vld1q_u8(qh +  0), vbit), m0);
    ثابت uint8x16_t b1   = vandq_u8(vtstq_u8(vld1q_u8(qh + 16), vbit), m1);
    ثابت uint8x16_t b01  = vorrq_u8(b0, b1);
    ثابت uint8x8_t  fold = vorr_u8(vget_کم_u8(b01), vget_بالا_u8(b01));
    ثابت uint8x8_t  بیرون  = vpadd_u8(fold, fold);
    plane[0 * 32 + ردیف] = vget_lane_u8(بیرون, 0);
    plane[1 * 32 + ردیف] = vget_lane_u8(بیرون, 1);
    plane[2 * 32 + ردیف] = vget_lane_u8(بیرون, 2);
    plane[3 * 32 + ردیف] = vget_lane_u8(بیرون, 3);
}

ساكن مضمن uint16x8_t repack_neon_بارگیری_f16x8(ثابت uint16_t * src) {
    أرجع vld1q_u16(src);
}

#إذا defined(__aarch64__) || defined(__ARM_FP16_FORMAT_IEEE)
ساكن مضمن uint16x8_t repack_neon_بارگیری_f16x8(ثابت عائم * src) {
    ثابت float16x4_t lo = vcvt_f16_f32(vld1q_f32(src));
    ثابت float16x4_t hi = vcvt_f16_f32(vld1q_f32(src + 4));
    أرجع vreinterpretq_u16_f16(vcombine_f16(lo, hi));
}

ساكن مضمن uint16x8_t repack_neon_بارگیری_f16x8(ثابت ggml_bf16_t * src) {
    ثابت uint16x8_t  bf   = vld1q_u16((ثابت uint16_t *) src);
    ثابت float32x4_t f_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_کم_u16(bf), 16));
    ثابت float32x4_t f_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_بالا_u16(bf), 16));
    ثابت float16x4_t lo   = vcvt_f16_f32(f_lo);
    ثابت float16x4_t hi   = vcvt_f16_f32(f_hi);
    أرجع vreinterpretq_u16_f16(vcombine_f16(lo, hi));
}
#وإلا
ساكن مضمن uint16x8_t repack_neon_بارگیری_f16x8(ثابت عائم * src) {
    uint16_t tmp[8];
    لكل (صحيح i = 0; i < 8; ++i) {
        tmp[i] = GGML_FP32_مقصد_FP16(src[i]);
    }
    أرجع vld1q_u16(tmp);
}

ساكن مضمن uint16x8_t repack_neon_بارگیری_f16x8(ثابت ggml_bf16_t * src) {
    uint16_t tmp[8];
    لكل (صحيح i = 0; i < 8; ++i) {
        tmp[i] = GGML_FP32_مقصد_FP16(GGML_BF16_مقصد_FP32(src[i]));
    }
    أرجع vld1q_u16(tmp);
}
#پایان_اگر

قالب <نام_نوع SrcT>
ساكن فراغ repack_neon_f16_tile(uint16_t * tile_dst, ثابت SrcT * src_برش, int64_t ردیف_stride, صحيح ct, صحيح kt) {
    لكل (صحيح row0 = 0; row0 < 32; row0 += 4) {
        ثابت SrcT * r0 = src_برش + (int64_t) (ct * 32 + row0 + 0) * ردیف_stride + kt * 32;
        ثابت SrcT * r1 = src_برش + (int64_t) (ct * 32 + row0 + 1) * ردیف_stride + kt * 32;
        ثابت SrcT * r2 = src_برش + (int64_t) (ct * 32 + row0 + 2) * ردیف_stride + kt * 32;
        ثابت SrcT * r3 = src_برش + (int64_t) (ct * 32 + row0 + 3) * ردیف_stride + kt * 32;
        لكل (صحيح cp = 0; cp < 16; cp += 4) {
            ثابت uint32x4_t v0 = vreinterpretq_u32_u16(repack_neon_بارگیری_f16x8(r0 + 2 * cp));
            ثابت uint32x4_t v1 = vreinterpretq_u32_u16(repack_neon_بارگیری_f16x8(r1 + 2 * cp));
            ثابت uint32x4_t v2 = vreinterpretq_u32_u16(repack_neon_بارگیری_f16x8(r2 + 2 * cp));
            ثابت uint32x4_t v3 = vreinterpretq_u32_u16(repack_neon_بارگیری_f16x8(r3 + 2 * cp));

            ثابت uint32x2x2_t t0 = vtrn_u32(vget_کم_u32(v0),  vget_کم_u32(v1));
            ثابت uint32x2x2_t t1 = vtrn_u32(vget_بالا_u32(v0), vget_بالا_u32(v1));
            ثابت uint32x2x2_t t2 = vtrn_u32(vget_کم_u32(v2),  vget_کم_u32(v3));
            ثابت uint32x2x2_t t3 = vtrn_u32(vget_بالا_u32(v2), vget_بالا_u32(v3));

            vst1q_u16(tile_dst + (cp + 0) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t0.val[0], t2.val[0])));
            vst1q_u16(tile_dst + (cp + 1) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t0.val[1], t2.val[1])));
            vst1q_u16(tile_dst + (cp + 2) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t1.val[0], t3.val[0])));
            vst1q_u16(tile_dst + (cp + 3) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t1.val[1], t3.val[1])));
        }
    }
}
#پایان_اگر

// --- Generic 32x32 tile grid walkers for block-32 and float types ---

قالب <نام_نوع Fn>
ساكن فراغ repack_parallel_for_ct(size_t برش_بایتها, صحيح n_ستون_tiles, Fn && fn) {
    صحيح n_نخها = 1;
    إذا (برش_بایتها > 128 * 1024) {
        لكل (صحيح nt : { 4, 3, 2 }) {
            إذا (n_ستون_tiles % nt == 0) {
                n_نخها = nt;
                اكسر;
            }
        }
    }

    إذا (n_نخها <= 1) {
        fn(0, n_ستون_tiles);
        أرجع;
    }

    ثابت صحيح chunk = n_ستون_tiles / n_نخها;
    std::thread workers[3];
    لكل (صحيح i = 0; i < n_نخها - 1; ++i) {
        workers[i] = std::thread([&fn, i, chunk]() {
            fn(i * chunk, (i + 1) * chunk);
        });
    }
    fn((n_نخها - 1) * chunk, n_ستون_tiles);
    لكل (صحيح i = 0; i < n_نخها - 1; ++i) {
        workers[i].join();
    }
}

قالب <نام_نوع SrcElemT, صحيح ELEMS_در_هر_KTILE, نام_نوع PackTileFn>
ساكن فراغ repack_مقصد_tiled_grid_32(ggml_tensor * t, ثابت فراغ * داده, size_t جابه_جایی, size_t size,
                                    size_t tile_اندازه, PackTileFn && pack_tile) {
    ثابت SrcElemT * src_ماتریس = (ثابت SrcElemT *) داده;
    ثابت int64_t ne0 = t->ne[0];
    ثابت int64_t ne1 = t->ne[1];
    ثابت int64_t ne2 = t->ne[2];
    ثابت int64_t ne3 = t->ne[3];
    ثابت صحيح n_ستون_tiles = hex_دور_به_بالا(ne1, 32) / 32;
    ثابت صحيح n_k_tiles   = hex_دور_به_بالا(ne0, 32) / 32;
    ثابت size_t ماتریس_اندازه = (size_t) n_ستون_tiles * n_k_tiles * tile_اندازه;

    ثابت size_t برش_اندازه = (size_t) ne1 * ggml_ردیف_اندازه(t->type, ne0);
    ثابت int64_t شروع_برش = جابه_جایی / برش_اندازه;
    int64_t انتها_برش = (جابه_جایی + size + برش_اندازه - 1) / برش_اندازه;
    إذا (انتها_برش > ne2 * ne3) {
        انتها_برش = ne2 * ne3;
    }

    ثابت int64_t ردیف_stride = (ne0 / (32 / ELEMS_در_هر_KTILE));

    لكل (int64_t برش_idx = شروع_برش; برش_idx < انتها_برش; برش_idx++) {
        ثابت SrcElemT * src_برش = src_ماتریس + (برش_idx - شروع_برش) * (ne1 * ردیف_stride);
        uint8_t * ماتریس_dst = (uint8_t *) t->داده + برش_idx * ماتریس_اندازه;

        repack_parallel_for_ct(ماتریس_اندازه, n_ستون_tiles, [&](صحيح ct0, صحيح ct1) {
            لكل (صحيح ct = ct0; ct < ct1; ct++) {
                لكل (صحيح kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = ماتریس_dst + ((size_t) ct * n_k_tiles + kt) * tile_اندازه;
                    pack_tile(tile_dst, src_برش, ne0, ne1, ردیف_stride, ct, kt);
                }
            }
        });
    }
}

قالب <نام_نوع DstElemT, صحيح ELEMS_در_هر_KTILE, نام_نوع UnpackTileFn>
ساكن فراغ unpack_from_tiled_grid_32(فراغ * داده, ثابت ggml_tensor * t, size_t جابه_جایی, size_t size,
                                      size_t tile_اندازه, UnpackTileFn && unpack_tile) {
    uint8_t * dst_بایتها = (uint8_t *) داده;
    ثابت int64_t ne0 = t->ne[0];
    ثابت int64_t ne1 = t->ne[1];
    ثابت int64_t ne2 = t->ne[2];
    ثابت int64_t ne3 = t->ne[3];
    ثابت صحيح n_ستون_tiles = hex_دور_به_بالا(ne1, 32) / 32;
    ثابت صحيح n_k_tiles   = hex_دور_به_بالا(ne0, 32) / 32;
    ثابت size_t ماتریس_اندازه = (size_t) n_ستون_tiles * n_k_tiles * tile_اندازه;

    ثابت size_t ردیف_اندازه_بایتها = ggml_ردیف_اندازه(t->type, ne0);
    ثابت size_t برش_اندازه     = (size_t) ne1 * ردیف_اندازه_بایتها;
    ثابت int64_t شروع_برش   = جابه_جایی / برش_اندازه;
    int64_t انتها_برش           = (جابه_جایی + size + برش_اندازه - 1) / برش_اندازه;
    إذا (انتها_برش > ne2 * ne3) {
        انتها_برش = ne2 * ne3;
    }

    ثابت int64_t ردیف_stride = (ne0 / (32 / ELEMS_در_هر_KTILE));

    لكل (int64_t برش_idx = شروع_برش; برش_idx < انتها_برش; برش_idx++) {
        ثابت size_t cur_شروع_byte     = (std::max)(جابه_جایی, (size_t) برش_idx * برش_اندازه);
        ثابت size_t cur_انتها_byte       = (std::min)(جابه_جایی + size, (size_t) (برش_idx + 1) * برش_اندازه);
        ثابت size_t برش_جابه_جایی_شروع = cur_شروع_byte - (size_t) برش_idx * برش_اندازه;
        ثابت size_t برش_جابه_جایی_انتها   = cur_انتها_byte - (size_t) برش_idx * برش_اندازه;

        ثابت int64_t شروع_ردیف = برش_جابه_جایی_شروع / ردیف_اندازه_بایتها;
        ثابت int64_t انتها_ردیف   = (std::min)((int64_t) ((برش_جابه_جایی_انتها + ردیف_اندازه_بایتها - 1) / ردیف_اندازه_بایتها), ne1);

        ثابت صحيح شروع_ct = (صحيح) (شروع_ردیف / 32);
        ثابت صحيح انتها_ct   = (std::min)((صحيح) ((انتها_ردیف + 31) / 32), n_ستون_tiles);

        DstElemT * dst_برش = (DstElemT *) (dst_بایتها + (cur_شروع_byte - جابه_جایی));
        ثابت uint8_t * ماتریس_src = (ثابت uint8_t *) t->داده + برش_idx * ماتریس_اندازه;

        لكل (صحيح ct = شروع_ct; ct < انتها_ct; ct++) {
            لكل (صحيح kt = 0; kt < n_k_tiles; kt++) {
                ثابت uint8_t * tile_src = ماتریس_src + ((size_t) ct * n_k_tiles + kt) * tile_اندازه;
                unpack_tile(dst_برش, tile_src, ne0, شروع_ردیف, انتها_ردیف, ردیف_stride, ct, kt);
            }
        }
    }
}

// --- Block-32 Quantized Types (Q4_0, IQ4_NL, Q4_1, Q8_0, MXFP4) ---

ساكن فراغ repack_q4_0_tiled(ggml_tensor * t, ثابت فراغ * داده, size_t جابه_جایی, size_t size) {
    repack_مقصد_tiled_grid_32<بلوک_q4_0, 1>(t, داده, جابه_جایی, size, HTP_TENSOR_TILE_اندازه_Q4_0,
        [](uint8_t * tile_dst, ثابت بلوک_q4_0 * src_برش, int64_t ne0, int64_t ne1, int64_t ردیف_stride, صحيح ct, صحيح kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
#إذا defined(__ARM_NEON) || defined(__aarch64__)
            إذا (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                لكل (صحيح row0 = 0; row0 < 32; row0 += 8) {
                    ثابت uint8_t * qs[8];
                    لكل (صحيح rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_برش[(ct * 32 + row0 + rr) * ردیف_stride + kt].qs;
                    }
                    repack_neon_4bit_ردیفها(qs, tile_dst, row0);
                }

                لكل (صحيح ردیف = 0; ردیف < 32; ++ردیف) {
                    scale_dst[ردیف] = src_برش[(ct * 32 + ردیف) * ردیف_stride + kt].d;
                }
                أرجع;
            }
#پایان_اگر
            uint8_t tile_quants[32][32];
            لكل (صحيح ردیف = 0; ردیف < 32; ردیف++) {
                ثابت int64_t r = ct * 32 + ردیف;
                إذا (r < ne1 && kt < ne0 / 32) {
                    ثابت بلوک_q4_0 & b = src_برش[r * ردیف_stride + kt];
                    unpack_nibbles_32(tile_quants[ردیف], b.qs);
                    scale_dst[ردیف] = b.d;
                } وإلا {
                    memset(tile_quants[ردیف], 8, 32);
                    scale_dst[ردیف] = 0;
                }
            }
            transpose_nibbles_مقصد_tile(tile_dst, tile_quants);
        });
}

ساكن فراغ repack_tiled_q4_0(فراغ * داده, ثابت ggml_tensor * t, size_t جابه_جایی, size_t size) {
    unpack_from_tiled_grid_32<بلوک_q4_0, 1>(داده, t, جابه_جایی, size, HTP_TENSOR_TILE_اندازه_Q4_0,
        [](بلوک_q4_0 * dst_برش, ثابت uint8_t * tile_src, int64_t ne0, int64_t شروع_ردیف, int64_t انتها_ردیف, int64_t ردیف_stride, صحيح ct, صحيح kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_مقصد_nibbles(tile_quants, tile_src);
            ثابت ggml_half * scale_src = (ثابت ggml_half *) (tile_src + 512);
            لكل (صحيح ردیف = 0; ردیف < 32; ردیف++) {
                ثابت int64_t r = ct * 32 + ردیف;
                إذا (r >= شروع_ردیف && r < انتها_ردیف && kt < ne0 / 32) {
                    بلوک_q4_0 & b = dst_برش[(r - شروع_ردیف) * ردیف_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[ردیف]);
                    b.d = scale_src[ردیف];
                }
            }
        });
}

ساكن فراغ repack_q4_1_tiled(ggml_tensor * t, ثابت فراغ * داده, size_t جابه_جایی, size_t size) {
    repack_مقصد_tiled_grid_32<بلوک_q4_1, 1>(t, داده, جابه_جایی, size, HTP_TENSOR_TILE_اندازه_Q4_1,
        [](uint8_t * tile_dst, ثابت بلوک_q4_1 * src_برش, int64_t ne0, int64_t ne1, int64_t ردیف_stride, صحيح ct, صحيح kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
#إذا defined(__ARM_NEON) || defined(__aarch64__)
            إذا (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                لكل (صحيح row0 = 0; row0 < 32; row0 += 8) {
                    ثابت uint8_t * qs[8];
                    لكل (صحيح rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_برش[(ct * 32 + row0 + rr) * ردیف_stride + kt].qs;
                    }
                    repack_neon_4bit_ردیفها(qs, tile_dst, row0);
                }

                لكل (صحيح ردیف = 0; ردیف < 32; ++ردیف) {
                    ثابت بلوک_q4_1 & src_بلوک = src_برش[(ct * 32 + ردیف) * ردیف_stride + kt];
                    scale_dst[2 * ردیف + 0] = src_بلوک.d;
                    scale_dst[2 * ردیف + 1] = src_بلوک.m;
                }
                أرجع;
            }
#پایان_اگر
            uint8_t tile_quants[32][32];
            لكل (صحيح ردیف = 0; ردیف < 32; ردیف++) {
                ثابت int64_t r = ct * 32 + ردیف;
                إذا (r < ne1 && kt < ne0 / 32) {
                    ثابت بلوک_q4_1 & b = src_برش[r * ردیف_stride + kt];
                    unpack_nibbles_32(tile_quants[ردیف], b.qs);
                    scale_dst[2 * ردیف + 0] = b.d;
                    scale_dst[2 * ردیف + 1] = b.m;
                } وإلا {
                    memset(tile_quants[ردیف], 0, 32);
                    scale_dst[2 * ردیف + 0] = 0;
                    scale_dst[2 * ردیف + 1] = 0;
                }
            }
            transpose_nibbles_مقصد_tile(tile_dst, tile_quants);
        });
}

ساكن فراغ repack_tiled_q4_1(فراغ * داده, ثابت ggml_tensor * t, size_t جابه_جایی, size_t size) {
    unpack_from_tiled_grid_32<بلوک_q4_1, 1>(داده, t, جابه_جایی, size, HTP_TENSOR_TILE_اندازه_Q4_1,
        [](بلوک_q4_1 * dst_برش, ثابت uint8_t * tile_src, int64_t ne0, int64_t شروع_ردیف, int64_t انتها_ردیف, int64_t ردیف_stride, صحيح ct, صحيح kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_مقصد_nibbles(tile_quants, tile_src);
            ثابت ggml_half * scale_src = (ثابت ggml_half *) (tile_src + 512);
            لكل (صحيح ردیف = 0; ردیف < 32; ردیف++) {
                ثابت int64_t r = ct * 32 + ردیف;
                إذا (r >= شروع_ردیف && r < انتها_ردیف && kt < ne0 / 32) {
                    بلوک_q4_1 & b = dst_برش[(r - شروع_ردیف) * ردیف_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[ردیف]);
                    b.d = scale_src[2 * ردیف + 0];
                    b.m = scale_src[2 * ردیف + 1];
                }
            }
        });
}

ساكن فراغ repack_mxfp4_tiled(ggml_tensor * t, ثابت فراغ * داده, size_t جابه_جایی, size_t size) {
    repack_مقصد_tiled_grid_32<بلوک_mxfp4, 1>(t, داده, جابه_جایی, size, HTP_TENSOR_TILE_اندازه_MXFP4,
        [](uint8_t * tile_dst, ثابت بلوک_mxfp4 * src_برش, int64_t ne0, int64_t ne1, int64_t ردیف_stride, صحيح ct, صحيح kt) {
            uint8_t * scale_dst = tile_dst + 512;
#إذا defined(__ARM_NEON) || defined(__aarch64__)
            إذا (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                لكل (صحيح row0 = 0; row0 < 32; row0 += 8) {
                    ثابت uint8_t * qs[8];
                    لكل (صحيح rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_برش[(ct * 32 + row0 + rr) * ردیف_stride + kt].qs;
                    }
                    repack_neon_4bit_ردیفها(qs, tile_dst, row0);
                }

                لكل (صحيح ردیف = 0; ردیف < 32; ++ردیف) {
                    scale_dst[ردیف] = src_برش[(ct * 32 + ردیف) * ردیف_stride + kt].e;
                }
                أرجع;
            }
#پایان_اگر
            uint8_t tile_quants[32][32];
            لكل (صحيح ردیف = 0; ردیف < 32; ردیف++) {
                ثابت int64_t r = ct * 32 + ردیف;
                إذا (r < ne1 && kt < ne0 / 32) {
                    ثابت بلوک_mxfp4 & b = src_برش[r * ردیف_stride + kt];
                    unpack_nibbles_32(tile_quants[ردیف], b.qs);
                    scale_dst[ردیف] = b.e;
                } وإلا {
                    memset(tile_quants[ردیف], 0, 32);
                    scale_dst[ردیف] = 0;
                }
            }
            transpose_nibbles_مقصد_tile(tile_dst, tile_quants);
        });
}

ساكن فراغ repack_tiled_mxfp4(فراغ * داده, ثابت ggml_tensor * t, size_t جابه_جایی, size_t size) {
    unpack_from_tiled_grid_32<بلوک_mxfp4, 1>(داده, t, جابه_جایی, size, HTP_TENSOR_TILE_اندازه_MXFP4,
        [](بلوک_mxfp4 * dst_برش, ثابت uint8_t * tile_src, int64_t ne0, int64_t شروع_ردیف, int64_t انتها_ردیف, int64_t ردیف_stride, صحيح ct, صحيح kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_مقصد_nibbles(tile_quants, tile_src);
            ثابت uint8_t * scale_src = tile_src + 512;
            لكل (صحيح ردیف = 0; ردیف < 32; ردیف++) {
                ثابت int64_t r = ct * 32 + ردیف;
                إذا (r >= شروع_ردیف && r < انتها_ردیف && kt < ne0 / 32) {
                    بلوک_mxfp4 & b = dst_برش[(r - شروع_ردیف) * ردیف_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[ردیف]);
                    b.e = scale_src[ردیف];
                }
            }
        });
}

ساكن فراغ repack_q8_0_tiled(ggml_tensor * t, ثابت فراغ * داده, size_t جابه_جایی, size_t size) {
    repack_مقصد_tiled_grid_32<بلوک_q8_0, 1>(t, داده, جابه_جایی, size, HTP_TENSOR_TILE_اندازه_Q8_0,
        [](uint8_t * tile_dst, ثابت بلوک_q8_0 * src_برش, int64_t ne0, int64_t ne1, int64_t ردیف_stride, صحيح ct, صحيح kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 1024);
#إذا defined(__ARM_NEON) || defined(__aarch64__)
            إذا (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                لكل (صحيح row0 = 0; row0 < 32; row0 += 8) {
                    ثابت uint8_t * qs[8];
                    لكل (صحيح rr = 0; rr < 8; ++rr) {
                        qs[rr] = (ثابت uint8_t *) src_برش[(ct * 32 + row0 + rr) * ردیف_stride + kt].qs;
                    }
                    repack_neon_q8_ردیفها(qs, tile_dst, row0);
                }

                لكل (صحيح ردیف = 0; ردیف < 32; ++ردیف) {
                    scale_dst[ردیف] = src_برش[(ct * 32 + ردیف) * ردیف_stride + kt].d;
                }
                أرجع;
            }
#پایان_اگر
            لكل (صحيح cp = 0; cp < 16; cp++) {
                ثابت صحيح col0 = cp * 2;
                ثابت صحيح col1 = col0 + 1;
                لكل (صحيح ردیف = 0; ردیف < 32; ردیف++) {
                    ثابت int64_t r = ct * 32 + ردیف;
                    ثابت بلوک_q8_0 * b = (r < ne1 && kt < ne0 / 32) ? &src_برش[r * ردیف_stride + kt] : اشاره_تهی;
                    tile_dst[cp * 64 + 2 * ردیف + 0] = b ? b->qs[col0] : 0;
                    tile_dst[cp * 64 + 2 * ردیف + 1] = b ? b->qs[col1] : 0;
                }
            }
            لكل (صحيح ردیف = 0; ردیف < 32; ردیف++) {
                ثابت int64_t r = ct * 32 + ردیف;
                scale_dst[ردیف] = (r < ne1 && kt < ne0 / 32) ? src_برش[r * ردیف_stride + kt].d : 0;
            }
        });
}

ساكن فراغ repack_tiled_q8_0(فراغ * داده, ثابت ggml_tensor * t, size_t جابه_جایی, size_t size) {
    unpack_from_tiled_grid_32<بلوک_q8_0, 1>(داده, t, جابه_جایی, size, HTP_TENSOR_TILE_اندازه_Q8_0,
        [](بلوک_q8_0 * dst_برش, ثابت uint8_t * tile_src, int64_t ne0, int64_t شروع_ردیف, int64_t انتها_ردیف, int64_t ردیف_stride, صحيح ct, صحيح kt) {
            لكل (صحيح cp = 0; cp < 16; cp++) {
                ثابت صحيح col0 = cp * 2;
                ثابت صحيح col1 = col0 + 1;
                لكل (صحيح ردیف = 0; ردیف < 32; ردیف++) {
                    ثابت int64_t r = ct * 32 + ردیف;
                    إذا (r >= شروع_ردیف && r < انتها_ردیف && kt < ne0 / 32) {
                        بلوک_q8_0 & b = dst_برش[(r - شروع_ردیف) * ردیف_stride + kt];
                        b.qs[col0] = tile_src[cp * 64 + 2 * ردیف + 0];
                        b.qs[col1] = tile_src[cp * 64 + 2 * ردیف + 1];
                    }
                }
            }
            ثابت ggml_half * scale_src = (ثابت ggml_half *) (tile_src + 1024);
            لكل (صحيح ردیف = 0; ردیف < 32; ردیف++) {
                ثابت int64_t r = ct * 32 + ردیف;
                إذا (r >= شروع_ردیف && r < انتها_ردیف && kt < ne0 / 32) {
                    dst_برش[(r - شروع_ردیف) * ردیف_stride + kt].d = scale_src[ردیف];
                }
            }
        });
}

// --- Float Types (F16, F32, BF16 -> f16_tiled HMX layout) ---

ساكن فراغ repack_q2_0_tiled(ggml_tensor * t, ثابت فراغ * داده, size_t جابه_جایی, size_t size) {
    GGML_ASSERT(جابه_جایی == 0);
    GGML_ASSERT(t->ne[0] % 64 == 0);
    GGML_ASSERT(size == ggml_nbytes(t));

    ثابت بلوک_q2_0 * src = (ثابت بلوک_q2_0 *) داده;
    ثابت int64_t بلوکها_در_هر_ردیف = t->ne[0] / 64;
    ثابت int64_t n_ستون_tiles = hex_دور_به_بالا(t->ne[1], 32) / 32;
    ثابت int64_t n_وزن_tiles = بلوکها_در_هر_ردیف;
    ثابت size_t ماتریس_اندازه = (size_t) n_ستون_tiles * n_وزن_tiles * HTP_TENSOR_TILE_اندازه_Q2_0;

    لكل (int64_t برش_2 = 0; برش_2 < t->ne[2] * t->ne[3]; ++برش_2) {
        ثابت بلوک_q2_0 * src_برش = src + برش_2 * t->ne[1] * بلوکها_در_هر_ردیف;
        uint8_t * dst_برش = (uint8_t *) t->داده + برش_2 * ماتریس_اندازه;
        ثابت size_t ct_بایتها = (size_t) n_وزن_tiles * HTP_TENSOR_TILE_اندازه_Q2_0;

        repack_parallel_for_ct(ماتریس_اندازه, (صحيح) n_ستون_tiles, [&](صحيح ct0, صحيح ct1) {
            memset(dst_برش + (size_t) ct0 * ct_بایتها, 0, (size_t) (ct1 - ct0) * ct_بایتها);
            ثابت int64_t r0 = (int64_t) ct0 * 32;
            ثابت int64_t r1 = (std::min)((int64_t) ct1 * 32, t->ne[1]);
            لكل (int64_t ردیف_شاخص = r0; ردیف_شاخص < r1; ++ردیف_شاخص) {
                ثابت int64_t ct = ردیف_شاخص / 32;
                ثابت صحيح ردیف = (صحيح) (ردیف_شاخص % 32);
                لكل (int64_t kt = 0; kt < n_وزن_tiles; ++kt) {
                    ثابت بلوک_q2_0 * بلوک = src_برش + ردیف_شاخص * بلوکها_در_هر_ردیف + kt;
                    uint8_t * tile = dst_برش + (ct * n_وزن_tiles + kt) * HTP_TENSOR_TILE_اندازه_Q2_0;
#إذا defined(__ARM_NEON) || defined(__aarch64__)
                    repack_neon_q2_0_بلوک(بلوک->qs, tile, ردیف);
#وإلا
                    لكل (صحيح گروه = 0; گروه < 16; ++گروه) {
                        لكل (صحيح byte = 0; byte < 4; ++byte) {
                            ثابت صحيح k = 4 * گروه + byte;
                            ثابت uint8_t code = (بلوک->qs[k / 4] >> (2 * (k % 4))) & 3;
                            tile[(گروه >> 2) * 128 + 4 * ردیف + byte] |= code << (2 * (گروه & 3));
                        }
                    }
#پایان_اگر
                    کپی_حافظه(tile + 512 + 2 * ردیف, &بلوک->d, حجم(بلوک->d));
                }
            }
        });
    }
}

ساكن فراغ repack_tiled_q2_0(فراغ * داده, ثابت ggml_tensor * t, size_t جابه_جایی, size_t size) {
    GGML_ASSERT(t->ne[0] % 64 == 0);
    GGML_ASSERT(جابه_جایی + size <= ggml_nbytes(t));

    ثابت size_t اولی_بلوک = جابه_جایی / حجم(بلوک_q2_0);
    ثابت size_t انتها_بلوک = (جابه_جایی + size + حجم(بلوک_q2_0) - 1) / حجم(بلوک_q2_0);
    ثابت size_t بلوکها_در_هر_ردیف = t->ne[0] / 64;
    ثابت size_t بلوکها_در_هر_برش = t->ne[1] * بلوکها_در_هر_ردیف;
    ثابت size_t n_وزن_tiles = بلوکها_در_هر_ردیف;
    ثابت size_t n_ستون_tiles = hex_دور_به_بالا(t->ne[1], 32) / 32;
    ثابت size_t ماتریس_اندازه = n_ستون_tiles * n_وزن_tiles * HTP_TENSOR_TILE_اندازه_Q2_0;

    لكل (size_t بلوک_شاخص = اولی_بلوک; بلوک_شاخص < انتها_بلوک; ++بلوک_شاخص) {
        ثابت size_t برش_2 = بلوک_شاخص / بلوکها_در_هر_برش;
        ثابت size_t درون_برش = بلوک_شاخص % بلوکها_در_هر_برش;
        ثابت size_t ردیف_شاخص = درون_برش / بلوکها_در_هر_ردیف;
        ثابت size_t kt = درون_برش % بلوکها_در_هر_ردیف;
        ثابت size_t ct = ردیف_شاخص / 32;
        ثابت size_t ردیف = ردیف_شاخص % 32;
        ثابت uint8_t * tile = (ثابت uint8_t *) t->داده + برش_2 * ماتریس_اندازه +
            (ct * n_وزن_tiles + kt) * HTP_TENSOR_TILE_اندازه_Q2_0;

        بلوک_q2_0 بلوک = {};
        کپی_حافظه(&بلوک.d, tile + 512 + 2 * ردیف, حجم(بلوک.d));
        لكل (صحيح گروه = 0; گروه < 16; ++گروه) {
            لكل (صحيح byte = 0; byte < 4; ++byte) {
                ثابت uint8_t packed = tile[(گروه >> 2) * 128 + 4 * ردیف + byte];
                ثابت uint8_t code = (packed >> (2 * (گروه & 3))) & 3;
                بلوک.qs[گروه] |= code << (2 * byte);
            }
        }

        ثابت size_t بلوک_begin = بلوک_شاخص * حجم(بلوک_q2_0);
        ثابت size_t copy_begin = (std::max)(بلوک_begin, جابه_جایی);
        ثابت size_t copy_انتها = (std::min)(بلوک_begin + حجم(بلوک_q2_0), جابه_جایی + size);
        کپی_حافظه((uint8_t *) داده + copy_begin - جابه_جایی, (ثابت uint8_t *) &بلوک + copy_begin - بلوک_begin, copy_انتها - copy_begin);
    }
}

قالب <نام_نوع SrcT, نام_نوع مقصد_Fp16_Fn>
ساكن فراغ repack_float_مقصد_f16_tiled(ggml_tensor * t, ثابت فراغ * داده, size_t جابه_جایی, size_t size, مقصد_Fp16_Fn && مقصد_fp16) {
    repack_مقصد_tiled_grid_32<SrcT, 32>(t, داده, جابه_جایی, size, HTP_TENSOR_TILE_اندازه_F16,
        [&](uint8_t * tile_بایتها, ثابت SrcT * src_برش, int64_t ne0, int64_t ne1, int64_t ردیف_stride, صحيح ct, صحيح kt) {
            uint16_t * tile_dst = (uint16_t *) tile_بایتها;
#إذا defined(__ARM_NEON) || defined(__aarch64__)
            إذا (ct * 32 + 32 <= ne1 && kt * 32 + 32 <= ne0) {
                repack_neon_f16_tile(tile_dst, src_برش, ردیف_stride, ct, kt);
                أرجع;
            }
#پایان_اگر
            لكل (صحيح cp = 0; cp < 16; cp++) {
                ثابت int64_t c0 = kt * 32 + cp * 2;
                ثابت int64_t c1 = c0 + 1;
                لكل (صحيح ردیف = 0; ردیف < 32; ردیف++) {
                    ثابت int64_t r = ct * 32 + ردیف;
                    ثابت SrcT * r_ptr = (r < ne1) ? &src_برش[r * ردیف_stride] : اشاره_تهی;
                    tile_dst[cp * 64 + ردیف * 2 + 0] = (r_ptr && c0 < ne0) ? مقصد_fp16(r_ptr[c0]) : 0;
                    tile_dst[cp * 64 + ردیف * 2 + 1] = (r_ptr && c1 < ne0) ? مقصد_fp16(r_ptr[c1]) : 0;
                }
            }
        });
}

قالب <نام_نوع DstT, نام_نوع FromFp16Fn>
ساكن فراغ unpack_f16_tiled_مقصد_float(فراغ * داده, ثابت ggml_tensor * t, size_t جابه_جایی, size_t size, FromFp16Fn && from_fp16) {
    unpack_from_tiled_grid_32<DstT, 32>(داده, t, جابه_جایی, size, HTP_TENSOR_TILE_اندازه_F16,
        [&](DstT * dst_برش, ثابت uint8_t * tile_بایتها, int64_t ne0, int64_t شروع_ردیف, int64_t انتها_ردیف, int64_t ردیف_stride, صحيح ct, صحيح kt) {
            ثابت uint16_t * tile_src = (ثابت uint16_t *) tile_بایتها;
            لكل (صحيح cp = 0; cp < 16; cp++) {
                ثابت int64_t c0 = kt * 32 + cp * 2;
                ثابت int64_t c1 = c0 + 1;
                لكل (صحيح ردیف = 0; ردیف < 32; ردیف++) {
                    ثابت int64_t r = ct * 32 + ردیف;
                    إذا (r >= شروع_ردیف && r < انتها_ردیف) {
                        DstT * r_ptr = dst_برش + (r - شروع_ردیف) * ردیف_stride;
                        إذا (c0 < ne0) r_ptr[c0] = from_fp16(tile_src[cp * 64 + ردیف * 2 + 0]);
                        إذا (c1 < ne0) r_ptr[c1] = from_fp16(tile_src[cp * 64 + ردیف * 2 + 1]);
                    }
                }
            }
        });
}

// --- Generic K-Quant (super-block 256) grid walkers ---

قالب <نام_نوع بلوک_T, نام_نوع PackKtFn>
ساكن فراغ repack_kquant_مقصد_tiled_grid(ggml_tensor * t, ثابت فراغ * داده, size_t جابه_جایی, size_t size,
                                        size_t tile_اندازه, PackKtFn && pack_kt) {
    GGML_ASSERT(جابه_جایی == 0);
    GGML_استفادهنشده(size);

    ثابت بلوک_T * src_ماتریس = (ثابت بلوک_T *) داده;
    ثابت int64_t ne0 = t->ne[0];
    ثابت int64_t ne1 = t->ne[1];
    ثابت int64_t ne2 = t->ne[2];
    ثابت int64_t ne3 = t->ne[3];
    GGML_ASSERT(ne0 % QK_K == 0);

    ثابت صحيح n_ستون_tiles    = hex_دور_به_بالا(ne1, 32) / 32;
    ثابت صحيح n_k_tiles      = hex_دور_به_بالا(ne0, 32) / 32;
    ثابت size_t ماتریس_اندازه = (size_t) n_ستون_tiles * n_k_tiles * tile_اندازه;
    ثابت int64_t sb_در_هر_ردیف = ne0 / QK_K;

    لكل (int64_t برش_idx = 0; برش_idx < ne2 * ne3; برش_idx++) {
        ثابت بلوک_T * src_برش = src_ماتریس + برش_idx * (ne1 * sb_در_هر_ردیف);
        uint8_t * ماتریس_dst     = (uint8_t *) t->داده + برش_idx * ماتریس_اندازه;
        ثابت size_t ct_بایتها    = (size_t) n_k_tiles * tile_اندازه;

        repack_parallel_for_ct(ماتریس_اندازه, n_ستون_tiles, [&](صحيح ct0, صحيح ct1) {
            // Padding rows and OR-ed bit planes require zeroed tiles
            memset(ماتریس_dst + (size_t) ct0 * ct_بایتها, 0, (size_t) (ct1 - ct0) * ct_بایتها);

            ثابت int64_t r0 = (int64_t) ct0 * 32;
            ثابت int64_t r1 = (std::min)((int64_t) ct1 * 32, ne1);
            لكل (int64_t r = r0; r < r1; r++) {
                ثابت صحيح ct  = (صحيح) (r / 32);
                ثابت صحيح ردیف = (صحيح) (r % 32);
                ثابت بلوک_T * src_ردیف = src_برش + r * sb_در_هر_ردیف;

                لكل (صحيح kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile = ماتریس_dst + ((size_t) ct * n_k_tiles + kt) * tile_اندازه;
                    pack_kt(tile, &src_ردیف[kt / 8], ردیف, kt % 8);
                }
            }
        });
    }
}

قالب <نام_نوع بلوک_T, نام_نوع UnpackSbFn>
ساكن فراغ unpack_kquant_from_tiled_grid(فراغ * داده, ثابت ggml_tensor * t, size_t جابه_جایی, size_t size,
                                          size_t tile_اندازه, UnpackSbFn && unpack_sb) {
    uint8_t * dst_بایتها = (uint8_t *) داده;
    ثابت int64_t ne0 = t->ne[0];
    ثابت int64_t ne1 = t->ne[1];
    ثابت int64_t ne2 = t->ne[2];
    ثابت int64_t ne3 = t->ne[3];
    GGML_ASSERT(ne0 % QK_K == 0);

    ثابت صحيح n_ستون_tiles    = hex_دور_به_بالا(ne1, 32) / 32;
    ثابت صحيح n_k_tiles      = hex_دور_به_بالا(ne0, 32) / 32;
    ثابت size_t ماتریس_اندازه = (size_t) n_ستون_tiles * n_k_tiles * tile_اندازه;
    ثابت int64_t sb_در_هر_ردیف = ne0 / QK_K;

    ثابت size_t ردیف_اندازه_بایتها = ggml_ردیف_اندازه(t->type, ne0);
    ثابت size_t برش_اندازه     = (size_t) ne1 * ردیف_اندازه_بایتها;
    ثابت int64_t شروع_برش   = جابه_جایی / برش_اندازه;
    int64_t انتها_برش           = (جابه_جایی + size + برش_اندازه - 1) / برش_اندازه;
    إذا (انتها_برش > ne2 * ne3) {
        انتها_برش = ne2 * ne3;
    }

    لكل (int64_t برش_idx = شروع_برش; برش_idx < انتها_برش; برش_idx++) {
        ثابت size_t cur_شروع_byte     = (std::max)(جابه_جایی, (size_t) برش_idx * برش_اندازه);
        ثابت size_t cur_انتها_byte       = (std::min)(جابه_جایی + size, (size_t) (برش_idx + 1) * برش_اندازه);
        ثابت size_t برش_جابه_جایی_شروع = cur_شروع_byte - (size_t) برش_idx * برش_اندازه;
        ثابت size_t برش_جابه_جایی_انتها   = cur_انتها_byte - (size_t) برش_idx * برش_اندازه;

        ثابت int64_t شروع_ردیف = برش_جابه_جایی_شروع / ردیف_اندازه_بایتها;
        ثابت int64_t انتها_ردیف   = (std::min)((int64_t) ((برش_جابه_جایی_انتها + ردیف_اندازه_بایتها - 1) / ردیف_اندازه_بایتها), ne1);

        بلوک_T * dst_برش         = (بلوک_T *) (dst_بایتها + (cur_شروع_byte - جابه_جایی));
        ثابت uint8_t * ماتریس_src = (ثابت uint8_t *) t->داده + برش_idx * ماتریس_اندازه;

        لكل (int64_t r = شروع_ردیف; r < انتها_ردیف; r++) {
            ثابت صحيح ct  = (صحيح) (r / 32);
            ثابت صحيح ردیف = (صحيح) (r % 32);
            بلوک_T * dst_ردیف = dst_برش + (r - شروع_ردیف) * sb_در_هر_ردیف;

            لكل (int64_t sb = 0; sb < sb_در_هر_ردیف; sb++) {
                بلوک_T * b = &dst_ردیف[sb];
                memset(b, 0, حجم(بلوک_T));
                ثابت uint8_t * sb_tiles = ماتریس_src + ((size_t) ct * n_k_tiles + sb * 8) * tile_اندازه;
                unpack_sb(b, sb_tiles, tile_اندازه, ردیف);
            }
        }
    }
}

// --- K-Quant Helpers & Per-Type Implementations (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) ---

#إذا !defined(__ARM_NEON) && !defined(__aarch64__)
// unsigned 6-bit value (0..63) of element e of a Q6_K block, same bit layout as dequantize_row_q6_K
ساكن مضمن uint8_t q6_K_get_quant(ثابت بلوک_q6_K * b, صحيح e) {
    ثابت صحيح c = e / 128;
    ثابت صحيح w = e % 128;
    ثابت صحيح g = w / 32;
    ثابت صحيح l = w % 32;
    ثابت uint8_t * ql = b->ql + c * 64;
    ثابت uint8_t * qh = b->qh + c * 32;
    uint8_t lo, hi;
    بدل (g) {
        حالة 0:  lo = ql[l]      & 0xF; hi = (qh[l] >> 0) & 3; اكسر;
        حالة 1:  lo = ql[l + 32] & 0xF; hi = (qh[l] >> 2) & 3; اكسر;
        حالة 2:  lo = ql[l]      >> 4;  hi = (qh[l] >> 4) & 3; اكسر;
        افتراضي: lo = ql[l + 32] >> 4;  hi = (qh[l] >> 6) & 3; اكسر;
    }
    أرجع (uint8_t) (lo | (hi << 4));
}
#پایان_اگر

ساكن فراغ repack_q6_K_tiled(ggml_tensor * t, ثابت فراغ * داده, size_t جابه_جایی, size_t size) {
    repack_kquant_مقصد_tiled_grid<بلوک_q6_K>(t, داده, جابه_جایی, size, HTP_TENSOR_TILE_اندازه_Q6_K,
        [](uint8_t * tile, ثابت بلوک_q6_K * b, صحيح ردیف, صحيح kt_local) {
            ثابت عائم d     = GGML_FP16_مقصد_FP32(b->d);
            uint8_t * lo_pl   = tile;
            uint8_t * hi_pl   = tile + 512;
            ggml_half * sc_pl = (ggml_half *) (tile + 768);

#إذا defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q6_k(b, kt_local, lo_pl, hi_pl, ردیف);
#وإلا
            لكل (صحيح lk = 0; lk < 32; lk++) {
                ثابت uint8_t q6 = q6_K_get_quant(b, kt_local * 32 + lk);
                ثابت صحيح g   = lk >> 2;
                ثابت صحيح pos = ردیف * 4 + (lk & 3);
                lo_pl[(g >> 1) * 128 + pos] |= (uint8_t) ((q6 & 0xF) << ((g & 1) * 4));
                hi_pl[(g >> 2) * 128 + pos] |= (uint8_t) ((q6 >> 4) << ((g & 3) * 2));
            }
#پایان_اگر
            لكل (صحيح sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + ردیف] = GGML_FP32_مقصد_FP16(d * (عائم) b->scales[kt_local * 2 + sub]);
            }
        });
}

ساكن فراغ repack_tiled_q6_K(فراغ * داده, ثابت ggml_tensor * t, size_t جابه_جایی, size_t size) {
    unpack_kquant_from_tiled_grid<بلوک_q6_K>(داده, t, جابه_جایی, size, HTP_TENSOR_TILE_اندازه_Q6_K,
        [](بلوک_q6_K * b, ثابت uint8_t * sb_tiles, size_t tile_اندازه, صحيح ردیف) {
            عائم sub_scales[16];
            لكل (صحيح kt_local = 0; kt_local < 8; kt_local++) {
                ثابت uint8_t *   tile  = sb_tiles + kt_local * tile_اندازه;
                ثابت uint8_t *   lo_pl = tile;
                ثابت uint8_t *   hi_pl = tile + 512;
                ثابت ggml_half * sc_pl = (ثابت ggml_half *) (tile + 768);

                ثابت صحيح c  = kt_local / 4;
                ثابت صحيح g  = kt_local % 4;
                uint8_t * ql = b->ql + c * 64;
                uint8_t * qh = b->qh + c * 32;

                لكل (صحيح lk = 0; lk < 32; lk++) {
                    ثابت صحيح g_tile = lk >> 2;
                    ثابت صحيح pos    = ردیف * 4 + (lk & 3);
                    ثابت uint8_t lo = (lo_pl[(g_tile >> 1) * 128 + pos] >> ((g_tile & 1) * 4)) & 0xF;
                    ثابت uint8_t hi = (hi_pl[(g_tile >> 2) * 128 + pos] >> ((g_tile & 3) * 2)) & 3;

                    بدل (g) {
                        حالة 0:  ql[lk]      |= lo;        qh[lk] |= (hi << 0); اكسر;
                        حالة 1:  ql[lk + 32] |= lo;        qh[lk] |= (hi << 2); اكسر;
                        حالة 2:  ql[lk]      |= (lo << 4); qh[lk] |= (hi << 4); اكسر;
                        افتراضي: ql[lk + 32] |= (lo << 4); qh[lk] |= (hi << 6); اكسر;
                    }
                }

                لكل (صحيح sub = 0; sub < 2; sub++) {
                    sub_scales[kt_local * 2 + sub] = GGML_FP16_مقصد_FP32(sc_pl[sub * 32 + ردیف]);
                }
            }

            عائم max_abs_scale = 0.0f;
            لكل (صحيح s = 0; s < 16; s++) {
                max_abs_scale = (std::max)(max_abs_scale, fabsf(sub_scales[s]));
            }

            إذا (max_abs_scale == 0.0f) {
                b->d = GGML_FP32_مقصد_FP16(0.0f);
            } وإلا {
                b->d = GGML_FP32_مقصد_FP16(max_abs_scale / 127.0f);
                ثابت عائم d_actual = GGML_FP16_مقصد_FP32(b->d);
                ثابت عائم inv_d    = (d_actual != 0.0f) ? (1.0f / d_actual) : 0.0f;
                لكل (صحيح s = 0; s < 16; s++) {
                    ثابت صحيح sc = (صحيح) roundf(sub_scales[s] * inv_d);
                    b->scales[s] = (int8_t) (std::max)(-128, (std::min)(127, sc));
                }
            }
        });
}

#إذا !defined(__ARM_NEON) && !defined(__aarch64__)
// low 2 bits (0..3) of element e of a Q2_K or Q3_K block, same bit layout as dequantize_row_q2_K / q3_K
ساكن مضمن uint8_t q2_3_K_get_low2(ثابت uint8_t * qs, صحيح e) {
    ثابت صحيح c = e / 128;
    ثابت صحيح j = (e % 128) / 32;
    ثابت صحيح l = e % 32;
    أرجع (qs[c * 32 + l] >> (2 * j)) & 3;
}

// hmask bit of element e of a Q3_K block, same bit layout as dequantize_row_q3_K
ساكن مضمن منطقي q3_K_get_hbit(ثابت بلوک_q3_K * b, صحيح e) {
    ثابت صحيح c = e / 128;
    ثابت صحيح j = (e % 128) / 32;
    ثابت صحيح l = e % 32;
    أرجع (b->hmask[l] >> (c * 4 + j)) & 1;
}
#پایان_اگر

// signed 6-bit scale j (-32..31) of a Q3_K block, same packing as quantize_row_q3_K_ref
ساكن مضمن صحيح q3_K_get_scale(ثابت uint8_t * scales, صحيح j) {
    ثابت صحيح lo = (j < 8) ? (scales[j] & 0xF) : (scales[j - 8] >> 4);
    ثابت صحيح hi = (scales[8 + j % 4] >> (2 * (j / 4))) & 3;
    أرجع (lo | (hi << 4)) - 32;
}

// read-back: find fp16 d and l[j] in [lmin, lmax] with fp16(d * l[j]) == prod[j] for all j, false if none
ساكن منطقي hexagon_recover_k_scales(ثابت ggml_half * prod, صحيح n, صحيح lmin, صحيح lmax, ggml_half * d_بیرون, صحيح * l_بیرون) {
    صحيح jmax = 0;
    لكل (صحيح j = 1; j < n; j++) {
        إذا (fabsf(GGML_FP16_مقصد_FP32(prod[j])) > fabsf(GGML_FP16_مقصد_FP32(prod[jmax]))) {
            jmax = j;
        }
    }
    ثابت عائم pmax = GGML_FP16_مقصد_FP32(prod[jmax]);
    إذا (pmax == 0.0f) {
        *d_بیرون = GGML_FP32_مقصد_FP16(0.0f);
        لكل (صحيح j = 0; j < n; j++) {
            l_بیرون[j] = 0;
        }
        أرجع درست;
    }

    // the quantizers put the largest scale at or near the range end, so try large |l| first
    ثابت صحيح lext = (std::max)(-lmin, lmax);
    لكل (صحيح a = lext; a >= 1; a--) {
        لكل (صحيح sign : { 1, -1 }) {
            ثابت صحيح lj = sign * a;
            إذا (lj < lmin || lj > lmax) {
                استمر;
            }
            ثابت ggml_half d0 = GGML_FP32_مقصد_FP16(pmax / (عائم) lj);
            لكل (صحيح ulp : { 0, -1, 1 }) {
                ggml_half d = d0;
                uint16_t بیتها;
                کپی_حافظه(&بیتها, &d, حجم(بیتها));
                بیتها = (uint16_t) (بیتها + ulp);
                کپی_حافظه(&d, &بیتها, حجم(بیتها));

                ثابت عائم df = GGML_FP16_مقصد_FP32(d);
                إذا (!std::isfinite(df) || df == 0.0f) {
                    استمر;
                }
                منطقي ok = درست;
                لكل (صحيح j = 0; j < n && ok; j++) {
                    ثابت صحيح l = (صحيح) roundf(GGML_FP16_مقصد_FP32(prod[j]) / df);
                    ثابت ggml_half p = GGML_FP32_مقصد_FP16(df * (عائم) l);
                    ok = l >= lmin && l <= lmax && memcmp(&p, &prod[j], حجم(p)) == 0;
                    l_بیرون[j] = l;
                }
                إذا (ok) {
                    *d_بیرون = d;
                    أرجع درست;
                }
            }
        }
    }
    أرجع خطأ;
}

ساكن فراغ repack_q3_K_tiled(ggml_tensor * t, ثابت فراغ * داده, size_t جابه_جایی, size_t size) {
    repack_kquant_مقصد_tiled_grid<بلوک_q3_K>(t, داده, جابه_جایی, size, HTP_TENSOR_TILE_اندازه_Q3_K,
        [](uint8_t * tile, ثابت بلوک_q3_K * b, صحيح ردیف, صحيح kt_local) {
            ثابت عائم d     = GGML_FP16_مقصد_FP32(b->d);
            uint8_t * lo_pl   = tile;
            uint8_t * neg_pl  = tile + 256;
            ggml_half * sc_pl = (ggml_half *) (tile + 384);

#إذا defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q2_3_k_low2(b->qs, kt_local, lo_pl, ردیف);
            repack_neon_q3_k_hmask(b->hmask, kt_local, neg_pl, ردیف);
#وإلا
            لكل (صحيح lk = 0; lk < 32; lk++) {
                ثابت صحيح e   = kt_local * 32 + lk;
                ثابت صحيح g   = lk >> 2;
                ثابت صحيح pos = ردیف * 4 + (lk & 3);
                lo_pl[(g >> 2) * 128 + pos] |= (uint8_t) (q2_3_K_get_low2(b->qs, e) << ((g & 3) * 2));
                إذا (!q3_K_get_hbit(b, e)) {
                    neg_pl[pos] |= (uint8_t) (1 << g);
                }
            }
#پایان_اگر
            لكل (صحيح sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + ردیف] = GGML_FP32_مقصد_FP16(d * (عائم) q3_K_get_scale(b->scales, kt_local * 2 + sub));
            }
        });
}

ساكن فراغ repack_tiled_q3_K(فراغ * داده, ثابت ggml_tensor * t, size_t جابه_جایی, size_t size) {
    unpack_kquant_from_tiled_grid<بلوک_q3_K>(داده, t, جابه_جایی, size, HTP_TENSOR_TILE_اندازه_Q3_K,
        [](بلوک_q3_K * b, ثابت uint8_t * sb_tiles, size_t tile_اندازه, صحيح ردیف) {
            ggml_half sub_scales[16];
            لكل (صحيح kt_local = 0; kt_local < 8; kt_local++) {
                ثابت uint8_t *   tile   = sb_tiles + kt_local * tile_اندازه;
                ثابت uint8_t *   lo_pl  = tile;
                ثابت uint8_t *   neg_pl = tile + 256;
                ثابت ggml_half * sc_pl  = (ثابت ggml_half *) (tile + 384);

                لكل (صحيح lk = 0; lk < 32; lk++) {
                    ثابت صحيح e   = kt_local * 32 + lk;
                    ثابت صحيح g   = lk >> 2;
                    ثابت صحيح pos = ردیف * 4 + (lk & 3);
                    ثابت uint8_t lo = (lo_pl[(g >> 2) * 128 + pos] >> ((g & 3) * 2)) & 3;

                    ثابت صحيح c = e / 128;
                    ثابت صحيح j = (e % 128) / 32;
                    ثابت صحيح l = e % 32;
                    b->qs[c * 32 + l] |= (uint8_t) (lo << (2 * j));
                    إذا (!((neg_pl[pos] >> g) & 1)) {
                        b->hmask[l] |= (uint8_t) (1 << (c * 4 + j));
                    }
                }

                لكل (صحيح sub = 0; sub < 2; sub++) {
                    sub_scales[kt_local * 2 + sub] = sc_pl[sub * 32 + ردیف];
                }
            }

            صحيح ls[16];
            إذا (!hexagon_recover_k_scales(sub_scales, 16, -32, 31, &b->d, ls)) {
                عائم max_scale = 0.0f;
                لكل (صحيح s = 0; s < 16; s++) {
                    إذا (fabsf(GGML_FP16_مقصد_FP32(sub_scales[s])) > fabsf(max_scale)) {
                        max_scale = GGML_FP16_مقصد_FP32(sub_scales[s]);
                    }
                }
                b->d = GGML_FP32_مقصد_FP16(-max_scale / 32.0f);
                ثابت عائم d_actual = GGML_FP16_مقصد_FP32(b->d);
                ثابت عائم inv_d    = (d_actual != 0.0f) ? (1.0f / d_actual) : 0.0f;
                لكل (صحيح s = 0; s < 16; s++) {
                    ls[s] = (std::max)(-32, (std::min)(31, (صحيح) roundf(GGML_FP16_مقصد_FP32(sub_scales[s]) * inv_d)));
                }
            }

            لكل (صحيح s = 0; s < 16; s++) {
                ثابت صحيح l = ls[s] + 32;
                إذا (s < 8) {
                    b->scales[s] = l & 0xF;
                } وإلا {
                    b->scales[s - 8] |= (uint8_t) ((l & 0xF) << 4);
                }
                b->scales[s % 4 + 8] |= (uint8_t) ((l >> 4) << (2 * (s / 4)));
            }
        });
}

ساكن فراغ repack_q2_K_tiled(ggml_tensor * t, ثابت فراغ * داده, size_t جابه_جایی, size_t size) {
    repack_kquant_مقصد_tiled_grid<بلوک_q2_K>(t, داده, جابه_جایی, size, HTP_TENSOR_TILE_اندازه_Q2_K,
        [](uint8_t * tile, ثابت بلوک_q2_K * b, صحيح ردیف, صحيح kt_local) {
            uint8_t * lo_pl   = tile;
            uint8_t * sc_pl   = tile + 256;
            ggml_half * dd_pl = (ggml_half *) (tile + 320);

#إذا defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q2_3_k_low2(b->qs, kt_local, lo_pl, ردیف);
#وإلا
            لكل (صحيح lk = 0; lk < 32; lk++) {
                ثابت صحيح g   = lk >> 2;
                ثابت صحيح pos = ردیف * 4 + (lk & 3);
                lo_pl[(g >> 2) * 128 + pos] |= (uint8_t) (q2_3_K_get_low2(b->qs, kt_local * 32 + lk) << ((g & 3) * 2));
            }
#پایان_اگر
            لكل (صحيح sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + ردیف] = b->scales[kt_local * 2 + sub];
            }
            إذا (ردیف / 8 == kt_local % 4) {
                dd_pl[ردیف % 8] = kt_local < 4 ? b->d : (ggml_half) (b->dmin ^ 0x8000);
            }
        });
}

ساكن فراغ repack_tiled_q2_K(فراغ * داده, ثابت ggml_tensor * t, size_t جابه_جایی, size_t size) {
    unpack_kquant_from_tiled_grid<بلوک_q2_K>(داده, t, جابه_جایی, size, HTP_TENSOR_TILE_اندازه_Q2_K,
        [](بلوک_q2_K * b, ثابت uint8_t * sb_tiles, size_t tile_اندازه, صحيح ردیف) {
            لكل (صحيح kt_local = 0; kt_local < 8; kt_local++) {
                ثابت uint8_t *   tile  = sb_tiles + kt_local * tile_اندازه;
                ثابت uint8_t *   lo_pl = tile;
                ثابت uint8_t *   sc_pl = tile + 256;
                ثابت ggml_half * dd_pl = (ثابت ggml_half *) (tile + 320);

                لكل (صحيح lk = 0; lk < 32; lk++) {
                    ثابت صحيح e   = kt_local * 32 + lk;
                    ثابت صحيح g   = lk >> 2;
                    ثابت صحيح pos = ردیف * 4 + (lk & 3);
                    ثابت uint8_t lo = (lo_pl[(g >> 2) * 128 + pos] >> ((g & 3) * 2)) & 3;
                    b->qs[(e / 128) * 32 + e % 32] |= (uint8_t) (lo << (2 * ((e % 128) / 32)));
                }

                لكل (صحيح sub = 0; sub < 2; sub++) {
                    b->scales[kt_local * 2 + sub] = sc_pl[sub * 32 + ردیف];
                }
                إذا (ردیف / 8 == kt_local % 4) {
                    إذا (kt_local < 4) {
                        b->d = dd_pl[ردیف % 8];
                    } وإلا {
                        b->dmin = (ggml_half) (dd_pl[ردیف % 8] ^ 0x8000);
                    }
                }
            }
        });
}

ساكن مضمن فراغ get_scale_min_k4(صحيح j, ثابت uint8_t * q, uint8_t * d, uint8_t * m) {
    إذا (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } وإلا {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

ساكن مضمن فراغ pack_k4_scales_mins(uint8_t * scales, ggml_half * d, ggml_half * dmin,
                                       ثابت عائم * sub_scales, ثابت عائم * sub_mins) {
    عائم max_scale = 0.0f;
    عائم max_min   = 0.0f;
    لكل (صحيح j = 0; j < 8; j++) {
        إذا (sub_scales[j] > max_scale) max_scale = sub_scales[j];
        إذا (sub_mins[j]   > max_min)   max_min   = sub_mins[j];
    }

    عائم inv_scale = 0.0f;
    إذا (max_scale > 0.0f) {
        *d = GGML_FP32_مقصد_FP16(max_scale / 63.0f);
        ثابت عائم d_actual = GGML_FP16_مقصد_FP32(*d);
        inv_scale = (d_actual > 0.0f) ? (1.0f / d_actual) : 0.0f;
    } وإلا {
        *d = GGML_FP32_مقصد_FP16(0.0f);
    }

    عائم inv_min = 0.0f;
    إذا (max_min > 0.0f) {
        *dmin = GGML_FP32_مقصد_FP16(max_min / 63.0f);
        ثابت عائم dmin_actual = GGML_FP16_مقصد_FP32(*dmin);
        inv_min = (dmin_actual > 0.0f) ? (1.0f / dmin_actual) : 0.0f;
    } وإلا {
        *dmin = GGML_FP32_مقصد_FP16(0.0f);
    }

    لكل (صحيح j = 0; j < 8; j++) {
        uint8_t ls = (std::min)((uint8_t) 63, (uint8_t) roundf(inv_scale * sub_scales[j]));
        uint8_t lm = (std::min)((uint8_t) 63, (uint8_t) roundf(inv_min * sub_mins[j]));
        إذا (j < 4) {
            scales[j]     = ls;
            scales[j + 4] = lm;
        } وإلا {
            scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            scales[j - 4] |= ((ls >> 4) << 6);
            scales[j - 0] |= ((lm >> 4) << 6);
        }
    }
}

قالب <نام_نوع بلوک_T>
ساكن مضمن فراغ repack_k4_quants(uint8_t * tile_dst, ثابت بلوک_T * b, صحيح ردیف, صحيح kt_local,
                                    int64_t sb_در_هر_ردیف, منطقي use_neon) {
#إذا defined(__ARM_NEON) || defined(__aarch64__)
    إذا (use_neon) {
        إذا ((ردیف & 7) == 0) {
            ثابت uint8_t * qs[8];
            لكل (صحيح rr = 0; rr < 8; ++rr) {
                qs[rr] = b[rr * sb_در_هر_ردیف].qs + (kt_local / 2) * 32;
            }
            repack_neon_k4_ردیفها(qs, tile_dst, ردیف, (kt_local & 1) != 0);
        }
        أرجع;
    }
#وإلا
    GGML_استفادهنشده(sb_در_هر_ردیف);
    GGML_استفادهنشده(use_neon);
#پایان_اگر
    ثابت uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
    ثابت صحيح لغزش = (kt_local & 1) ? 4 : 0;

    لكل (صحيح cp = 0; cp < 16; cp++) {
        ثابت uint8_t q0 = (qs_sub[2 * cp + 0] >> لغزش) & 0x0F;
        ثابت uint8_t q1 = (qs_sub[2 * cp + 1] >> لغزش) & 0x0F;
        tile_dst[cp * 32 + ردیف] = (uint8_t) ((q1 << 4) | q0);
    }
}

ساكن فراغ repack_q4_K_tiled(ggml_tensor * t, ثابت فراغ * داده, size_t جابه_جایی, size_t size) {
    ثابت int64_t sb_در_هر_ردیف = t->ne[0] / QK_K;
    ثابت منطقي    use_neon   = (t->ne[1] % 8 == 0);
    repack_kquant_مقصد_tiled_grid<بلوک_q4_K>(t, داده, جابه_جایی, size, HTP_TENSOR_TILE_اندازه_Q4_K,
        [sb_در_هر_ردیف, use_neon](uint8_t * tile_dst, ثابت بلوک_q4_K * b, صحيح ردیف, صحيح kt_local) {
            ثابت عائم d    = GGML_FP16_مقصد_FP32(b->d);
            ثابت عائم dmin = GGML_FP16_مقصد_FP32(b->dmin);

            uint8_t sc, m;
            get_scale_min_k4(kt_local, b->scales, &sc, &m);

            repack_k4_quants(tile_dst, b, ردیف, kt_local, sb_در_هر_ردیف, use_neon);

            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
            scale_dst[2 * ردیف + 0] = GGML_FP32_مقصد_FP16(d * (عائم) sc);
            scale_dst[2 * ردیف + 1] = GGML_FP32_مقصد_FP16(-dmin * (عائم) m);
        });
}

ساكن فراغ repack_tiled_q4_K(فراغ * داده, ثابت ggml_tensor * t, size_t جابه_جایی, size_t size) {
    unpack_kquant_from_tiled_grid<بلوک_q4_K>(داده, t, جابه_جایی, size, HTP_TENSOR_TILE_اندازه_Q4_K,
        [](بلوک_q4_K * b, ثابت uint8_t * sb_tiles, size_t tile_اندازه, صحيح ردیف) {
            عائم sub_scales[8];
            عائم sub_mins[8];

            لكل (صحيح kt_local = 0; kt_local < 8; kt_local++) {
                ثابت uint8_t *   tile_src  = sb_tiles + kt_local * tile_اندازه;
                ثابت ggml_half * scale_src = (ثابت ggml_half *) (tile_src + 512);

                uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
                ثابت صحيح لغزش  = (kt_local & 1) ? 4 : 0;

                لكل (صحيح cp = 0; cp < 16; cp++) {
                    ثابت uint8_t val = tile_src[cp * 32 + ردیف];
                    qs_sub[2 * cp + 0] |= (uint8_t) ((val & 0x0F) << لغزش);
                    qs_sub[2 * cp + 1] |= (uint8_t) ((val >> 4)   << لغزش);
                }

                ثابت عائم D = GGML_FP16_مقصد_FP32(scale_src[2 * ردیف + 0]);
                ثابت عائم M = GGML_FP16_مقصد_FP32(scale_src[2 * ردیف + 1]);
                sub_scales[kt_local] = (D > 0.0f) ? D : 0.0f;
                sub_mins[kt_local]   = (-M > 0.0f) ? -M : 0.0f;
            }

            pack_k4_scales_mins(b->scales, &b->d, &b->dmin, sub_scales, sub_mins);
        });
}

ساكن فراغ repack_q5_K_tiled(ggml_tensor * t, ثابت فراغ * داده, size_t جابه_جایی, size_t size) {
    ثابت int64_t sb_در_هر_ردیف = t->ne[0] / QK_K;
    ثابت منطقي    use_neon   = (t->ne[1] % 8 == 0);
    repack_kquant_مقصد_tiled_grid<بلوک_q5_K>(t, داده, جابه_جایی, size, HTP_TENSOR_TILE_اندازه_Q5_K,
        [sb_در_هر_ردیف, use_neon](uint8_t * tile_dst, ثابت بلوک_q5_K * b, صحيح ردیف, صحيح kt_local) {
            ثابت عائم d     = GGML_FP16_مقصد_FP32(b->d);
            ثابت عائم dmin  = GGML_FP16_مقصد_FP32(b->dmin);
            uint8_t * plane   = tile_dst + 640;

            uint8_t sc, m;
            get_scale_min_k4(kt_local, b->scales, &sc, &m);

            repack_k4_quants(tile_dst, b, ردیف, kt_local, sb_در_هر_ردیف, use_neon);

            ثابت uint8_t hbit = (uint8_t) (1 << kt_local);
#إذا defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q5_k_qh(b->qh, hbit, plane, ردیف);
#وإلا
            لكل (صحيح cp = 0; cp < 16; cp++) {
                ثابت صحيح i    = cp / 4;
                ثابت صحيح lane = (cp % 4) * 32 + ردیف;
                إذا (b->qh[2 * cp + 0] & hbit) {
                    plane[lane] |= (uint8_t) (1 << (2 * i));
                }
                إذا (b->qh[2 * cp + 1] & hbit) {
                    plane[lane] |= (uint8_t) (1 << (2 * i + 1));
                }
            }
#پایان_اگر

            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
            scale_dst[2 * ردیف + 0] = GGML_FP32_مقصد_FP16(d * (عائم) sc);
            scale_dst[2 * ردیف + 1] = GGML_FP32_مقصد_FP16(-dmin * (عائم) m);
        });
}

ساكن فراغ repack_tiled_q5_K(فراغ * داده, ثابت ggml_tensor * t, size_t جابه_جایی, size_t size) {
    unpack_kquant_from_tiled_grid<بلوک_q5_K>(داده, t, جابه_جایی, size, HTP_TENSOR_TILE_اندازه_Q5_K,
        [](بلوک_q5_K * b, ثابت uint8_t * sb_tiles, size_t tile_اندازه, صحيح ردیف) {
            عائم sub_scales[8];
            عائم sub_mins[8];

            لكل (صحيح kt_local = 0; kt_local < 8; kt_local++) {
                ثابت uint8_t *   tile_src  = sb_tiles + kt_local * tile_اندازه;
                ثابت uint8_t *   plane     = tile_src + 640;
                ثابت ggml_half * scale_src = (ثابت ggml_half *) (tile_src + 512);

                uint8_t * qs_sub   = b->qs + (kt_local / 2) * 32;
                ثابت صحيح لغزش    = (kt_local & 1) ? 4 : 0;
                ثابت uint8_t hbit = (uint8_t) (1 << kt_local);

                لكل (صحيح cp = 0; cp < 16; cp++) {
                    ثابت uint8_t val = tile_src[cp * 32 + ردیف];
                    qs_sub[2 * cp + 0] |= (uint8_t) ((val & 0x0F) << لغزش);
                    qs_sub[2 * cp + 1] |= (uint8_t) ((val >> 4)   << لغزش);

                    ثابت صحيح i    = cp / 4;
                    ثابت صحيح lane = (cp % 4) * 32 + ردیف;
                    إذا (plane[lane] & (1 << (2 * i))) {
                        b->qh[2 * cp + 0] |= hbit;
                    }
                    إذا (plane[lane] & (1 << (2 * i + 1))) {
                        b->qh[2 * cp + 1] |= hbit;
                    }
                }

                ثابت عائم D = GGML_FP16_مقصد_FP32(scale_src[2 * ردیف + 0]);
                ثابت عائم M = GGML_FP16_مقصد_FP32(scale_src[2 * ردیف + 1]);
                sub_scales[kt_local] = (D > 0.0f) ? D : 0.0f;
                sub_mins[kt_local]   = (-M > 0.0f) ? -M : 0.0f;
            }

            pack_k4_scales_mins(b->scales, &b->d, &b->dmin, sub_scales, sub_mins);
        });
}

ساكن مضمن فراغ q1_0_tiled_بیت_location(صحيح k, صحيح ردیف, صحيح & byte_جابه_جایی, صحيح & بیت) {
    ثابت صحيح quarter = k / 32;
    ثابت صحيح kq      = k % 32;

    byte_جابه_جایی = quarter * 128 + 4 * ردیف + (kq % 4);
    بیت         = kq / 4;
}

ساكن فراغ repack_q1_0_tiled(ggml_tensor * t, ثابت فراغ * داده, size_t جابه_جایی, size_t size) {
    ثابت بلوک_q1_0 * src_ماتریس = (ثابت بلوک_q1_0 *) داده;
    ثابت int64_t ne0 = t->ne[0];
    ثابت int64_t ne1 = t->ne[1];
    ثابت int64_t ne2 = t->ne[2];
    ثابت int64_t ne3 = t->ne[3];
    ثابت صحيح n_ستون_tiles = hex_دور_به_بالا(ne1, 32) / 32;
    ثابت صحيح n_k_tiles = ne0 / QK1_0;
    ثابت size_t tile_اندازه = HTP_TENSOR_TILE_اندازه_Q1_0;
    ثابت size_t ماتریس_اندازه = (size_t) n_ستون_tiles * n_k_tiles * tile_اندازه;
    ثابت size_t برش_اندازه = ne1 * ggml_ردیف_اندازه(t->type, ne0);
    ثابت int64_t شروع_برش = جابه_جایی / برش_اندازه;
    ثابت int64_t انتها_برش = (std::min)((int64_t) ((جابه_جایی + size + برش_اندازه - 1) / برش_اندازه), ne2 * ne3);

    GGML_ASSERT(ne0 % QK1_0 == 0);

    لكل (int64_t برش_idx = شروع_برش; برش_idx < انتها_برش; برش_idx++) {
        ثابت بلوک_q1_0 * src_برش = src_ماتریس + (برش_idx - شروع_برش) * ne1 * n_k_tiles;
        uint8_t * ماتریس_dst = (uint8_t *) t->داده + برش_idx * ماتریس_اندازه;

        repack_parallel_for_ct(ماتریس_اندازه, n_ستون_tiles, [&](صحيح ct0, صحيح ct1) {
            لكل (صحيح ct = ct0; ct < ct1; ct++) {
                لكل (صحيح kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = ماتریس_dst + ((size_t) ct * n_k_tiles + kt) * tile_اندازه;
                    ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
                    memset(tile_dst, 0, tile_اندازه);

                    لكل (صحيح ردیف = 0; ردیف < 32; ردیف++) {
                        ثابت int64_t r = ct * 32 + ردیف;
                        إذا (r >= ne1) {
                            استمر;
                        }

                        ثابت بلوک_q1_0 & بلوک = src_برش[r * n_k_tiles + kt];
                        scale_dst[ردیف] = بلوک.d;
#إذا defined(__ARM_NEON) || defined(__aarch64__)
                        repack_neon_q1_0_بلوک(بلوک.qs, tile_dst, ردیف);
#وإلا
                        لكل (صحيح k = 0; k < QK1_0; k++) {
                            صحيح byte_جابه_جایی;
                            صحيح بیت;
                            q1_0_tiled_بیت_location(k, ردیف, byte_جابه_جایی, بیت);
                            إذا (بلوک.qs[k / 8] & (1u << (k % 8))) {
                                tile_dst[byte_جابه_جایی] |= (uint8_t) (1u << بیت);
                            }
                        }
#پایان_اگر
                    }
                }
            }
        });
    }
}

ساكن فراغ repack_tiled_q1_0(فراغ * داده, ثابت ggml_tensor * t, size_t جابه_جایی, size_t size) {
    بلوک_q1_0 * dst_ماتریس = (بلوک_q1_0 *) داده;
    ثابت int64_t ne0 = t->ne[0];
    ثابت int64_t ne1 = t->ne[1];
    ثابت int64_t ne2 = t->ne[2];
    ثابت int64_t ne3 = t->ne[3];
    ثابت صحيح n_ستون_tiles = hex_دور_به_بالا(ne1, 32) / 32;
    ثابت صحيح n_k_tiles = ne0 / QK1_0;
    ثابت size_t tile_اندازه = HTP_TENSOR_TILE_اندازه_Q1_0;
    ثابت size_t ماتریس_اندازه = (size_t) n_ستون_tiles * n_k_tiles * tile_اندازه;
    ثابت size_t ردیف_اندازه_بایتها = ggml_ردیف_اندازه(t->type, ne0);
    ثابت size_t برش_اندازه = ne1 * ردیف_اندازه_بایتها;
    ثابت int64_t شروع_برش = جابه_جایی / برش_اندازه;
    ثابت int64_t انتها_برش = (std::min)((int64_t) ((جابه_جایی + size + برش_اندازه - 1) / برش_اندازه), ne2 * ne3);

    GGML_ASSERT(ne0 % QK1_0 == 0);

    لكل (int64_t برش_idx = شروع_برش; برش_idx < انتها_برش; برش_idx++) {
        ثابت size_t cur_شروع_byte = (std::max)(جابه_جایی, (size_t) برش_idx * برش_اندازه);
        ثابت size_t cur_انتها_byte = (std::min)(جابه_جایی + size, (size_t) (برش_idx + 1) * برش_اندازه);
        ثابت int64_t شروع_ردیف = (cur_شروع_byte - (size_t) برش_idx * برش_اندازه) / ردیف_اندازه_بایتها;
        ثابت int64_t انتها_ردیف = (std::min)((int64_t) ((cur_انتها_byte - (size_t) برش_idx * برش_اندازه + ردیف_اندازه_بایتها - 1) / ردیف_اندازه_بایتها), ne1);
        بلوک_q1_0 * dst_برش = dst_ماتریس + (cur_شروع_byte - جابه_جایی) / حجم(بلوک_q1_0);
        ثابت uint8_t * ماتریس_src = (ثابت uint8_t *) t->داده + برش_idx * ماتریس_اندازه;

        لكل (صحيح ct = شروع_ردیف / 32; ct < (انتها_ردیف + 31) / 32; ct++) {
            لكل (صحيح kt = 0; kt < n_k_tiles; kt++) {
                ثابت uint8_t * tile_src = ماتریس_src + ((size_t) ct * n_k_tiles + kt) * tile_اندازه;
                ثابت ggml_half * scale_src = (ثابت ggml_half *) (tile_src + 512);

                لكل (صحيح ردیف = 0; ردیف < 32; ردیف++) {
                    ثابت int64_t r = ct * 32 + ردیف;
                    إذا (r < شروع_ردیف || r >= انتها_ردیف) {
                        استمر;
                    }

                    بلوک_q1_0 & بلوک = dst_برش[(r - شروع_ردیف) * n_k_tiles + kt];
                    memset(بلوک.qs, 0, حجم(بلوک.qs));
                    بلوک.d = scale_src[ردیف];
                    لكل (صحيح k = 0; k < QK1_0; k++) {
                        صحيح byte_جابه_جایی;
                        صحيح بیت;
                        q1_0_tiled_بیت_location(k, ردیف, byte_جابه_جایی, بیت);
                        إذا (tile_src[byte_جابه_جایی] & (1u << بیت)) {
                            بلوک.qs[k / 8] |= (uint8_t) (1u << (k % 8));
                        }
                    }
                }
            }
        }
    }
}

// --- Top-level Repack / Unpack Dispatchers ---

فراغ htp_repack_tensor_tiled(ggml_tensor * tensor, ثابت فراغ * داده, size_t size) {
    بدل (tensor->type) {
        حالة GGML_TYPE_Q1_0:
            repack_q1_0_tiled(tensor, داده, 0, size);
            اكسر;
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_IQ4_NL:
            repack_q4_0_tiled(tensor, داده, 0, size);
            اكسر;
        حالة GGML_TYPE_Q4_1:
            repack_q4_1_tiled(tensor, داده, 0, size);
            اكسر;
        حالة GGML_TYPE_Q8_0:
            repack_q8_0_tiled(tensor, داده, 0, size);
            اكسر;
        حالة GGML_TYPE_MXFP4:
            repack_mxfp4_tiled(tensor, داده, 0, size);
            اكسر;
        حالة GGML_TYPE_Q2_0:
            repack_q2_0_tiled(tensor, داده, 0, size);
            اكسر;
        حالة GGML_TYPE_Q2_K:
            repack_q2_K_tiled(tensor, داده, 0, size);
            اكسر;
        حالة GGML_TYPE_Q3_K:
            repack_q3_K_tiled(tensor, داده, 0, size);
            اكسر;
        حالة GGML_TYPE_Q4_K:
            repack_q4_K_tiled(tensor, داده, 0, size);
            اكسر;
        حالة GGML_TYPE_Q5_K:
            repack_q5_K_tiled(tensor, داده, 0, size);
            اكسر;
        حالة GGML_TYPE_Q6_K:
            repack_q6_K_tiled(tensor, داده, 0, size);
            اكسر;
        حالة GGML_TYPE_F16:
            repack_float_مقصد_f16_tiled<uint16_t>(tensor, داده, 0, size,
                [](uint16_t v) { أرجع v; });
            اكسر;
        حالة GGML_TYPE_F32:
            repack_float_مقصد_f16_tiled<عائم>(tensor, داده, 0, size,
                [](عائم v) { أرجع (uint16_t) GGML_FP32_مقصد_FP16(v); });
            اكسر;
        حالة GGML_TYPE_BF16:
            repack_float_مقصد_f16_tiled<ggml_bf16_t>(tensor, داده, 0, size,
                [](ggml_bf16_t v) { أرجع (uint16_t) GGML_FP32_مقصد_FP16(GGML_BF16_مقصد_FP32(v)); });
            اكسر;
        افتراضي:
            اكسر;
    }
}

فراغ htp_unpack_tensor_tiled(فراغ * داده, ثابت ggml_tensor * tensor, size_t جابه_جایی, size_t size) {
    GGML_ASSERT(جابه_جایی + size <= ggml_nbytes(tensor));

    بدل (tensor->type) {
        حالة GGML_TYPE_Q1_0:
            repack_tiled_q1_0(داده, tensor, جابه_جایی, size);
            اكسر;
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_IQ4_NL:
            repack_tiled_q4_0(داده, tensor, جابه_جایی, size);
            اكسر;
        حالة GGML_TYPE_Q4_1:
            repack_tiled_q4_1(داده, tensor, جابه_جایی, size);
            اكسر;
        حالة GGML_TYPE_Q8_0:
            repack_tiled_q8_0(داده, tensor, جابه_جایی, size);
            اكسر;
        حالة GGML_TYPE_MXFP4:
            repack_tiled_mxfp4(داده, tensor, جابه_جایی, size);
            اكسر;
        حالة GGML_TYPE_Q2_0:
            repack_tiled_q2_0(داده, tensor, جابه_جایی, size);
            اكسر;
        حالة GGML_TYPE_Q2_K:
            repack_tiled_q2_K(داده, tensor, جابه_جایی, size);
            اكسر;
        حالة GGML_TYPE_Q3_K:
            repack_tiled_q3_K(داده, tensor, جابه_جایی, size);
            اكسر;
        حالة GGML_TYPE_Q4_K:
            repack_tiled_q4_K(داده, tensor, جابه_جایی, size);
            اكسر;
        حالة GGML_TYPE_Q5_K:
            repack_tiled_q5_K(داده, tensor, جابه_جایی, size);
            اكسر;
        حالة GGML_TYPE_Q6_K:
            repack_tiled_q6_K(داده, tensor, جابه_جایی, size);
            اكسر;
        حالة GGML_TYPE_F16:
            unpack_f16_tiled_مقصد_float<uint16_t>(داده, tensor, جابه_جایی, size,
                [](uint16_t v) { أرجع v; });
            اكسر;
        حالة GGML_TYPE_F32:
            unpack_f16_tiled_مقصد_float<عائم>(داده, tensor, جابه_جایی, size,
                [](uint16_t v) { أرجع ggml_fp16_مقصد_fp32((ggml_fp16_t) v); });
            اكسر;
        حالة GGML_TYPE_BF16:
            unpack_f16_tiled_مقصد_float<ggml_bf16_t>(داده, tensor, جابه_جایی, size,
                [](uint16_t v) { أرجع ggml_fp32_مقصد_bf16(ggml_fp16_مقصد_fp32((ggml_fp16_t) v)); });
            اكسر;
        افتراضي:
            کپی_حافظه(داده, (ثابت حرف *) tensor->داده + جابه_جایی, size);
            اكسر;
    }
}
