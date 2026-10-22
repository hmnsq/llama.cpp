#ضمن "htp-repack.h"

#تعریف_کړه GGML_COMMON_IMPL_CPP
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
#د_که_پای

// --- 4-bit block-32 nibble helpers (shared by Q4_0, IQ4_NL, Q4_1, MXFP4) ---

ساكن مضمن فراغ unpack_nibbles_32(uint8_t * dst, ثابت uint8_t * qs) {
    لكل (بینښه صحيح i = 0; i < 16; ++i) {
        dst[i + 0]  = qs[i] & 0x0F;
        dst[i + 16] = qs[i] >> 4;
    }
}

ساكن مضمن فراغ pack_nibbles_32(uint8_t * qs, ثابت uint8_t * src) {
    لكل (بینښه صحيح i = 0; i < 16; ++i) {
        qs[i] = src[i + 0] | (src[i + 16] << 4);
    }
}

ساكن مضمن فراغ transpose_nibbles_منزل_tile(uint8_t * tile_dst, ثابت uint8_t tile_quants[32][32]) {
    لكل (صحيح cp = 0; cp < 16; cp++) {
        لكل (صحيح لیکه = 0; لیکه < 32; لیکه++) {
            tile_dst[cp * 32 + لیکه] = (tile_quants[لیکه][2 * cp + 1] << 4) | tile_quants[لیکه][2 * cp];
        }
    }
}

ساكن مضمن فراغ transpose_tile_منزل_nibbles(uint8_t tile_quants[32][32], ثابت uint8_t * tile_src) {
    لكل (صحيح cp = 0; cp < 16; cp++) {
        لكل (صحيح لیکه = 0; لیکه < 32; لیکه++) {
            ثابت uint8_t val = tile_src[cp * 32 + لیکه];
            tile_quants[لیکه][2 * cp + 0] = val & 0x0F;
            tile_quants[لیکه][2 * cp + 1] = val >> 4;
        }
    }
}

#إذا defined(__ARM_NEON) || defined(__aarch64__)
ساكن فراغ repack_neon_transpose8x8(ثابت uint8x8_t لیکې[8], uint8x8_t columns[8]) {
    ثابت uint8x8x2_t t0 = vtrn_u8(لیکې[0], لیکې[1]);
    ثابت uint8x8x2_t t1 = vtrn_u8(لیکې[2], لیکې[3]);
    ثابت uint8x8x2_t t2 = vtrn_u8(لیکې[4], لیکې[5]);
    ثابت uint8x8x2_t t3 = vtrn_u8(لیکې[6], لیکې[7]);
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

ساكن uint8x8_t repack_neon_pack_4bit_جوړې(uint8x8_t لومړی, uint8x8_t second, منطقي لوړ) {
    ثابت uint8x8x2_t separated = vuzp_u8(لومړی, second);
    ثابت uint8x8_t ماسک = vdup_n_u8(0x0F);
    uint8x8_t even = separated.val[0];
    uint8x8_t odd  = separated.val[1];

    إذا (لوړ) {
        even = vshr_n_u8(even, 4);
        odd  = vshr_n_u8(odd, 4);
    }

    even = vand_u8(even, ماسک);
    odd  = vand_u8(odd, ماسک);
    أرجع vorr_u8(vshl_n_u8(odd, 4), even);
}

ساكن فراغ repack_neon_4bit_لیکې(ثابت uint8_t * qs[8], uint8_t * tile_dst, صحيح row0) {
    uint8x8_t ټیټ_لیکې[8];
    uint8x8_t لوړ_لیکې[8];
    uint8x8_t columns[8];

    لكل (صحيح rr = 0; rr < 8; ++rr) {
        ثابت uint8x16_t packed = vld1q_u8(qs[rr]);
        ثابت uint8x8_t لومړی = vget_ټیټ_u8(packed);
        ثابت uint8x8_t second = vget_لوړ_u8(packed);
        ټیټ_لیکې[rr]  = repack_neon_pack_4bit_جوړې(لومړی, second, خطأ);
        لوړ_لیکې[rr] = repack_neon_pack_4bit_جوړې(لومړی, second, ریښتیا);
    }

    repack_neon_transpose8x8(ټیټ_لیکې, columns);
    لكل (صحيح cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + cp * 32 + row0, columns[cp]);
    }

    repack_neon_transpose8x8(لوړ_لیکې, columns);
    لكل (صحيح cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + (cp + 8) * 32 + row0, columns[cp]);
    }
}

ساكن فراغ repack_neon_k4_لیکې(ثابت uint8_t * qs[8], uint8_t * tile_dst, صحيح row0, منطقي لوړ) {
    uint8x8_t لومړی_لیکې[8];
    uint8x8_t دوهم_لیکې[8];
    uint8x8_t columns[8];

    لكل (صحيح rr = 0; rr < 8; ++rr) {
        لومړی_لیکې[rr] = repack_neon_pack_4bit_جوړې(vld1_u8(qs[rr]), vld1_u8(qs[rr] + 8), لوړ);
        دوهم_لیکې[rr] = repack_neon_pack_4bit_جوړې(vld1_u8(qs[rr] + 16), vld1_u8(qs[rr] + 24), لوړ);
    }

    repack_neon_transpose8x8(لومړی_لیکې, columns);
    لكل (صحيح cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + cp * 32 + row0, columns[cp]);
    }

    repack_neon_transpose8x8(دوهم_لیکې, columns);
    لكل (صحيح cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + (cp + 8) * 32 + row0, columns[cp]);
    }
}

ساكن فراغ repack_neon_q8_لیکې(ثابت uint8_t * qs[8], uint8_t * tile_dst, صحيح row0) {
    uint8x8_t لیکې[8];
    uint8x8_t columns[8];

    لكل (صحيح cp = 0; cp < 16; cp += 4) {
        لكل (صحيح rr = 0; rr < 8; ++rr) {
            لیکې[rr] = vld1_u8(qs[rr] + 2 * cp);
        }
        repack_neon_transpose8x8(لیکې, columns);

        لكل (صحيح pair = 0; pair < 4; ++pair) {
            ثابت uint8x8x2_t interleaved = vzip_u8(columns[2 * pair], columns[2 * pair + 1]);
            uint8_t * dst = tile_dst + (cp + pair) * 64 + 2 * row0;
            vst1_u8(dst, interleaved.val[0]);
            vst1_u8(dst + 8, interleaved.val[1]);
        }
    }
}

ساكن مضمن فراغ repack_store_u32(uint8_t * dst, uint32_t v) {
    حافظه_نقل(dst, &v, حجم(v));
}

ساكن مضمن uint32x4_t repack_neon_transpose_2bit_4x4(uint32x4_t w) {
    uint32x4_t t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 6)), vdupq_n_u32(0x00CC00CCu));
    w = veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 6)));
    t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 12)), vdupq_n_u32(0x0000F0F0u));
    أرجع veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 12)));
}

ساكن مضمن فراغ repack_neon_q2_0_بلاک(ثابت uint8_t * qs, uint8_t * tile, صحيح لیکه) {
    ثابت uint32x4_t w = repack_neon_transpose_2bit_4x4(vreinterpretq_u32_u8(vld1q_u8(qs)));
    repack_store_u32(tile + 0 * 128 + 4 * لیکه, vgetq_lane_u32(w, 0));
    repack_store_u32(tile + 1 * 128 + 4 * لیکه, vgetq_lane_u32(w, 1));
    repack_store_u32(tile + 2 * 128 + 4 * لیکه, vgetq_lane_u32(w, 2));
    repack_store_u32(tile + 3 * 128 + 4 * لیکه, vgetq_lane_u32(w, 3));
}

ساكن مضمن فراغ repack_neon_q1_0_بلاک(ثابت uint8_t * qs, uint8_t * tile, صحيح لیکه) {
    uint32x4_t w = vreinterpretq_u32_u8(vld1q_u8(qs));
    uint32x4_t t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 1)), vdupq_n_u32(0x22222222u));
    w = veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 1)));
    t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 3)), vdupq_n_u32(0x0A0A0A0Au));
    w = veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 3)));
    w = repack_neon_transpose_2bit_4x4(w);
    repack_store_u32(tile + 0 * 128 + 4 * لیکه, vgetq_lane_u32(w, 0));
    repack_store_u32(tile + 1 * 128 + 4 * لیکه, vgetq_lane_u32(w, 1));
    repack_store_u32(tile + 2 * 128 + 4 * لیکه, vgetq_lane_u32(w, 2));
    repack_store_u32(tile + 3 * 128 + 4 * لیکه, vgetq_lane_u32(w, 3));
}

ساكن مضمن فراغ repack_neon_pack_2bit_plane(uint8x16_t v0, uint8x16_t v1, uint8_t * dst, صحيح لیکه) {
    ثابت uint32x4x2_t z   = vuzpq_u32(vreinterpretq_u32_u8(v0), vreinterpretq_u32_u8(v1));
    ثابت uint32x4_t   p01 = vorrq_u32(z.val[0], vshlq_n_u32(z.val[1], 2));
    ثابت uint32x2x2_t u   = vuzp_u32(vget_ټیټ_u32(p01), vget_لوړ_u32(p01));
    ثابت uint32x2_t   بهر = vorr_u32(u.val[0], vshl_n_u32(u.val[1], 4));
    repack_store_u32(dst + 0 * 128 + لیکه * 4, vget_lane_u32(بهر, 0));
    repack_store_u32(dst + 1 * 128 + لیکه * 4, vget_lane_u32(بهر, 1));
}

ساكن مضمن فراغ repack_neon_q2_3_k_low2(ثابت uint8_t * qs, صحيح kt_local, uint8_t * lo_pl, صحيح لیکه) {
    ثابت uint8_t *  qs_sub = qs + (kt_local / 4) * 32;
    ثابت int8x16_t  ښويدنه  = vdupq_n_s8(-2 * (kt_local % 4));
    ثابت uint8x16_t ماسک   = vdupq_n_u8(0x03);
    ثابت uint8x16_t v0     = vandq_u8(vshlq_u8(vld1q_u8(qs_sub +  0), ښويدنه), ماسک);
    ثابت uint8x16_t v1     = vandq_u8(vshlq_u8(vld1q_u8(qs_sub + 16), ښويدنه), ماسک);
    repack_neon_pack_2bit_plane(v0, v1, lo_pl, لیکه);
}

ساكن مضمن فراغ repack_neon_q3_k_hmask(ثابت uint8_t * hmask, صحيح kt_local, uint8_t * neg_pl, صحيح لیکه) {
    ساكن ثابت uint32_t ماسک_ټکي[4] = { 0x01010101u, 0x02020202u, 0x04040404u, 0x08080808u };
    ثابت uint8x16_t m0   = vreinterpretq_u8_u32(vld1q_u32(ماسک_ټکي));
    ثابت uint8x16_t m1   = vshlq_n_u8(m0, 4);
    ثابت uint8x16_t hbit = vdupq_n_u8((uint8_t) (1u << kt_local));
    ثابت uint8x16_t b0   = vbicq_u8(m0, vtstq_u8(vld1q_u8(hmask +  0), hbit));
    ثابت uint8x16_t b1   = vbicq_u8(m1, vtstq_u8(vld1q_u8(hmask + 16), hbit));
    ثابت uint32x4_t b01  = vreinterpretq_u32_u8(vorrq_u8(b0, b1));
    ثابت uint32x2_t f0   = vorr_u32(vget_ټیټ_u32(b01), vget_لوړ_u32(b01));
    ثابت uint32x2_t f1   = vpadd_u32(f0, f0);
    repack_store_u32(neg_pl + لیکه * 4, vget_lane_u32(f1, 0));
}

ساكن مضمن فراغ repack_neon_q6_k(ثابت بلاک_q6_K * b, صحيح kt_local, uint8_t * lo_pl, uint8_t * hi_pl, صحيح لیکه) {
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
    repack_store_u32(lo_pl + 0 * 128 + لیکه * 4, vgetq_lane_u32(lout, 0));
    repack_store_u32(lo_pl + 1 * 128 + لیکه * 4, vgetq_lane_u32(lout, 1));
    repack_store_u32(lo_pl + 2 * 128 + لیکه * 4, vgetq_lane_u32(lout, 2));
    repack_store_u32(lo_pl + 3 * 128 + لیکه * 4, vgetq_lane_u32(lout, 3));

    ثابت uint8x16_t hi0 = vandq_u8(vshlq_u8(vld1q_u8(qh +  0), hshift), mask2);
    ثابت uint8x16_t hi1 = vandq_u8(vshlq_u8(vld1q_u8(qh + 16), hshift), mask2);
    repack_neon_pack_2bit_plane(hi0, hi1, hi_pl, لیکه);
}

ساكن مضمن فراغ repack_neon_q5_k_qh(ثابت uint8_t * qh, uint8_t hbit, uint8_t * plane, صحيح لیکه) {
    ساكن ثابت uint8_t ماسک_بایټونه[16] = { 1, 2, 1, 2, 1, 2, 1, 2, 4, 8, 4, 8, 4, 8, 4, 8 };
    ثابت uint8x16_t m0   = vld1q_u8(ماسک_بایټونه);
    ثابت uint8x16_t m1   = vshlq_n_u8(m0, 4);
    ثابت uint8x16_t vbit = vdupq_n_u8(hbit);
    ثابت uint8x16_t b0   = vandq_u8(vtstq_u8(vld1q_u8(qh +  0), vbit), m0);
    ثابت uint8x16_t b1   = vandq_u8(vtstq_u8(vld1q_u8(qh + 16), vbit), m1);
    ثابت uint8x16_t b01  = vorrq_u8(b0, b1);
    ثابت uint8x8_t  fold = vorr_u8(vget_ټیټ_u8(b01), vget_لوړ_u8(b01));
    ثابت uint8x8_t  بهر  = vpadd_u8(fold, fold);
    plane[0 * 32 + لیکه] = vget_lane_u8(بهر, 0);
    plane[1 * 32 + لیکه] = vget_lane_u8(بهر, 1);
    plane[2 * 32 + لیکه] = vget_lane_u8(بهر, 2);
    plane[3 * 32 + لیکه] = vget_lane_u8(بهر, 3);
}

ساكن مضمن uint16x8_t repack_neon_بارول_f16x8(ثابت uint16_t * src) {
    أرجع vld1q_u16(src);
}

#إذا defined(__aarch64__) || defined(__ARM_FP16_FORMAT_IEEE)
ساكن مضمن uint16x8_t repack_neon_بارول_f16x8(ثابت عائم * src) {
    ثابت float16x4_t lo = vcvt_f16_f32(vld1q_f32(src));
    ثابت float16x4_t hi = vcvt_f16_f32(vld1q_f32(src + 4));
    أرجع vreinterpretq_u16_f16(vcombine_f16(lo, hi));
}

ساكن مضمن uint16x8_t repack_neon_بارول_f16x8(ثابت ggml_bf16_t * src) {
    ثابت uint16x8_t  bf   = vld1q_u16((ثابت uint16_t *) src);
    ثابت float32x4_t f_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_ټیټ_u16(bf), 16));
    ثابت float32x4_t f_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_لوړ_u16(bf), 16));
    ثابت float16x4_t lo   = vcvt_f16_f32(f_lo);
    ثابت float16x4_t hi   = vcvt_f16_f32(f_hi);
    أرجع vreinterpretq_u16_f16(vcombine_f16(lo, hi));
}
#وإلا
ساكن مضمن uint16x8_t repack_neon_بارول_f16x8(ثابت عائم * src) {
    uint16_t tmp[8];
    لكل (صحيح i = 0; i < 8; ++i) {
        tmp[i] = GGML_FP32_منزل_FP16(src[i]);
    }
    أرجع vld1q_u16(tmp);
}

ساكن مضمن uint16x8_t repack_neon_بارول_f16x8(ثابت ggml_bf16_t * src) {
    uint16_t tmp[8];
    لكل (صحيح i = 0; i < 8; ++i) {
        tmp[i] = GGML_FP32_منزل_FP16(GGML_BF16_منزل_FP32(src[i]));
    }
    أرجع vld1q_u16(tmp);
}
#د_که_پای

کينډۍ <ډول_نوم SrcT>
ساكن فراغ repack_neon_f16_tile(uint16_t * tile_dst, ثابت SrcT * src_ټوټه, int64_t لیکه_stride, صحيح ct, صحيح kt) {
    لكل (صحيح row0 = 0; row0 < 32; row0 += 4) {
        ثابت SrcT * r0 = src_ټوټه + (int64_t) (ct * 32 + row0 + 0) * لیکه_stride + kt * 32;
        ثابت SrcT * r1 = src_ټوټه + (int64_t) (ct * 32 + row0 + 1) * لیکه_stride + kt * 32;
        ثابت SrcT * r2 = src_ټوټه + (int64_t) (ct * 32 + row0 + 2) * لیکه_stride + kt * 32;
        ثابت SrcT * r3 = src_ټوټه + (int64_t) (ct * 32 + row0 + 3) * لیکه_stride + kt * 32;
        لكل (صحيح cp = 0; cp < 16; cp += 4) {
            ثابت uint32x4_t v0 = vreinterpretq_u32_u16(repack_neon_بارول_f16x8(r0 + 2 * cp));
            ثابت uint32x4_t v1 = vreinterpretq_u32_u16(repack_neon_بارول_f16x8(r1 + 2 * cp));
            ثابت uint32x4_t v2 = vreinterpretq_u32_u16(repack_neon_بارول_f16x8(r2 + 2 * cp));
            ثابت uint32x4_t v3 = vreinterpretq_u32_u16(repack_neon_بارول_f16x8(r3 + 2 * cp));

            ثابت uint32x2x2_t t0 = vtrn_u32(vget_ټیټ_u32(v0),  vget_ټیټ_u32(v1));
            ثابت uint32x2x2_t t1 = vtrn_u32(vget_لوړ_u32(v0), vget_لوړ_u32(v1));
            ثابت uint32x2x2_t t2 = vtrn_u32(vget_ټیټ_u32(v2),  vget_ټیټ_u32(v3));
            ثابت uint32x2x2_t t3 = vtrn_u32(vget_لوړ_u32(v2), vget_لوړ_u32(v3));

            vst1q_u16(tile_dst + (cp + 0) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t0.val[0], t2.val[0])));
            vst1q_u16(tile_dst + (cp + 1) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t0.val[1], t2.val[1])));
            vst1q_u16(tile_dst + (cp + 2) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t1.val[0], t3.val[0])));
            vst1q_u16(tile_dst + (cp + 3) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t1.val[1], t3.val[1])));
        }
    }
}
#د_که_پای

// --- Generic 32x32 tile grid walkers for block-32 and float types ---

کينډۍ <ډول_نوم Fn>
ساكن فراغ repack_parallel_for_ct(size_t ټوټه_بایټونه, صحيح n_ستون_tiles, Fn && fn) {
    صحيح n_تارونه = 1;
    إذا (ټوټه_بایټونه > 128 * 1024) {
        لكل (صحيح nt : { 4, 3, 2 }) {
            إذا (n_ستون_tiles % nt == 0) {
                n_تارونه = nt;
                اكسر;
            }
        }
    }

    إذا (n_تارونه <= 1) {
        fn(0, n_ستون_tiles);
        أرجع;
    }

    ثابت صحيح chunk = n_ستون_tiles / n_تارونه;
    std::thread workers[3];
    لكل (صحيح i = 0; i < n_تارونه - 1; ++i) {
        workers[i] = std::thread([&fn, i, chunk]() {
            fn(i * chunk, (i + 1) * chunk);
        });
    }
    fn((n_تارونه - 1) * chunk, n_ستون_tiles);
    لكل (صحيح i = 0; i < n_تارونه - 1; ++i) {
        workers[i].join();
    }
}

کينډۍ <ډول_نوم SrcElemT, صحيح ELEMS_فی_KTILE, ډول_نوم PackTileFn>
ساكن فراغ repack_منزل_tiled_grid_32(ggml_tensor * t, ثابت فراغ * معلومات, size_t دننه_واټن, size_t size,
                                    size_t tile_اندازه, PackTileFn && pack_tile) {
    ثابت SrcElemT * src_ماتریکس = (ثابت SrcElemT *) معلومات;
    ثابت int64_t ne0 = t->ne[0];
    ثابت int64_t ne1 = t->ne[1];
    ثابت int64_t ne2 = t->ne[2];
    ثابت int64_t ne3 = t->ne[3];
    ثابت صحيح n_ستون_tiles = hex_پړاو_پورته(ne1, 32) / 32;
    ثابت صحيح n_k_tiles   = hex_پړاو_پورته(ne0, 32) / 32;
    ثابت size_t ماتریکس_اندازه = (size_t) n_ستون_tiles * n_k_tiles * tile_اندازه;

    ثابت size_t ټوټه_اندازه = (size_t) ne1 * ggml_لیکه_اندازه(t->type, ne0);
    ثابت int64_t پیل_ټوټه = دننه_واټن / ټوټه_اندازه;
    int64_t پای_ټکی_ټوټه = (دننه_واټن + size + ټوټه_اندازه - 1) / ټوټه_اندازه;
    إذا (پای_ټکی_ټوټه > ne2 * ne3) {
        پای_ټکی_ټوټه = ne2 * ne3;
    }

    ثابت int64_t لیکه_stride = (ne0 / (32 / ELEMS_فی_KTILE));

    لكل (int64_t ټوټه_idx = پیل_ټوټه; ټوټه_idx < پای_ټکی_ټوټه; ټوټه_idx++) {
        ثابت SrcElemT * src_ټوټه = src_ماتریکس + (ټوټه_idx - پیل_ټوټه) * (ne1 * لیکه_stride);
        uint8_t * ماتریکس_dst = (uint8_t *) t->معلومات + ټوټه_idx * ماتریکس_اندازه;

        repack_parallel_for_ct(ماتریکس_اندازه, n_ستون_tiles, [&](صحيح ct0, صحيح ct1) {
            لكل (صحيح ct = ct0; ct < ct1; ct++) {
                لكل (صحيح kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = ماتریکس_dst + ((size_t) ct * n_k_tiles + kt) * tile_اندازه;
                    pack_tile(tile_dst, src_ټوټه, ne0, ne1, لیکه_stride, ct, kt);
                }
            }
        });
    }
}

کينډۍ <ډول_نوم DstElemT, صحيح ELEMS_فی_KTILE, ډول_نوم UnpackTileFn>
ساكن فراغ unpack_from_tiled_grid_32(فراغ * معلومات, ثابت ggml_tensor * t, size_t دننه_واټن, size_t size,
                                      size_t tile_اندازه, UnpackTileFn && unpack_tile) {
    uint8_t * dst_بایټونه = (uint8_t *) معلومات;
    ثابت int64_t ne0 = t->ne[0];
    ثابت int64_t ne1 = t->ne[1];
    ثابت int64_t ne2 = t->ne[2];
    ثابت int64_t ne3 = t->ne[3];
    ثابت صحيح n_ستون_tiles = hex_پړاو_پورته(ne1, 32) / 32;
    ثابت صحيح n_k_tiles   = hex_پړاو_پورته(ne0, 32) / 32;
    ثابت size_t ماتریکس_اندازه = (size_t) n_ستون_tiles * n_k_tiles * tile_اندازه;

    ثابت size_t لیکه_اندازه_بایټونه = ggml_لیکه_اندازه(t->type, ne0);
    ثابت size_t ټوټه_اندازه     = (size_t) ne1 * لیکه_اندازه_بایټونه;
    ثابت int64_t پیل_ټوټه   = دننه_واټن / ټوټه_اندازه;
    int64_t پای_ټکی_ټوټه           = (دننه_واټن + size + ټوټه_اندازه - 1) / ټوټه_اندازه;
    إذا (پای_ټکی_ټوټه > ne2 * ne3) {
        پای_ټکی_ټوټه = ne2 * ne3;
    }

    ثابت int64_t لیکه_stride = (ne0 / (32 / ELEMS_فی_KTILE));

    لكل (int64_t ټوټه_idx = پیل_ټوټه; ټوټه_idx < پای_ټکی_ټوټه; ټوټه_idx++) {
        ثابت size_t cur_پیل_byte     = (std::max)(دننه_واټن, (size_t) ټوټه_idx * ټوټه_اندازه);
        ثابت size_t cur_پای_ټکی_byte       = (std::min)(دننه_واټن + size, (size_t) (ټوټه_idx + 1) * ټوټه_اندازه);
        ثابت size_t ټوټه_دننه_واټن_پیل = cur_پیل_byte - (size_t) ټوټه_idx * ټوټه_اندازه;
        ثابت size_t ټوټه_دننه_واټن_پای_ټکی   = cur_پای_ټکی_byte - (size_t) ټوټه_idx * ټوټه_اندازه;

        ثابت int64_t پیل_لیکه = ټوټه_دننه_واټن_پیل / لیکه_اندازه_بایټونه;
        ثابت int64_t پای_ټکی_لیکه   = (std::min)((int64_t) ((ټوټه_دننه_واټن_پای_ټکی + لیکه_اندازه_بایټونه - 1) / لیکه_اندازه_بایټونه), ne1);

        ثابت صحيح پیل_ct = (صحيح) (پیل_لیکه / 32);
        ثابت صحيح پای_ټکی_ct   = (std::min)((صحيح) ((پای_ټکی_لیکه + 31) / 32), n_ستون_tiles);

        DstElemT * dst_ټوټه = (DstElemT *) (dst_بایټونه + (cur_پیل_byte - دننه_واټن));
        ثابت uint8_t * ماتریکس_src = (ثابت uint8_t *) t->معلومات + ټوټه_idx * ماتریکس_اندازه;

        لكل (صحيح ct = پیل_ct; ct < پای_ټکی_ct; ct++) {
            لكل (صحيح kt = 0; kt < n_k_tiles; kt++) {
                ثابت uint8_t * tile_src = ماتریکس_src + ((size_t) ct * n_k_tiles + kt) * tile_اندازه;
                unpack_tile(dst_ټوټه, tile_src, ne0, پیل_لیکه, پای_ټکی_لیکه, لیکه_stride, ct, kt);
            }
        }
    }
}

// --- Block-32 Quantized Types (Q4_0, IQ4_NL, Q4_1, Q8_0, MXFP4) ---

ساكن فراغ repack_q4_0_tiled(ggml_tensor * t, ثابت فراغ * معلومات, size_t دننه_واټن, size_t size) {
    repack_منزل_tiled_grid_32<بلاک_q4_0, 1>(t, معلومات, دننه_واټن, size, HTP_TENSOR_TILE_اندازه_Q4_0,
        [](uint8_t * tile_dst, ثابت بلاک_q4_0 * src_ټوټه, int64_t ne0, int64_t ne1, int64_t لیکه_stride, صحيح ct, صحيح kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
#إذا defined(__ARM_NEON) || defined(__aarch64__)
            إذا (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                لكل (صحيح row0 = 0; row0 < 32; row0 += 8) {
                    ثابت uint8_t * qs[8];
                    لكل (صحيح rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_ټوټه[(ct * 32 + row0 + rr) * لیکه_stride + kt].qs;
                    }
                    repack_neon_4bit_لیکې(qs, tile_dst, row0);
                }

                لكل (صحيح لیکه = 0; لیکه < 32; ++لیکه) {
                    scale_dst[لیکه] = src_ټوټه[(ct * 32 + لیکه) * لیکه_stride + kt].d;
                }
                أرجع;
            }
#د_که_پای
            uint8_t tile_quants[32][32];
            لكل (صحيح لیکه = 0; لیکه < 32; لیکه++) {
                ثابت int64_t r = ct * 32 + لیکه;
                إذا (r < ne1 && kt < ne0 / 32) {
                    ثابت بلاک_q4_0 & b = src_ټوټه[r * لیکه_stride + kt];
                    unpack_nibbles_32(tile_quants[لیکه], b.qs);
                    scale_dst[لیکه] = b.d;
                } وإلا {
                    memset(tile_quants[لیکه], 8, 32);
                    scale_dst[لیکه] = 0;
                }
            }
            transpose_nibbles_منزل_tile(tile_dst, tile_quants);
        });
}

ساكن فراغ repack_tiled_q4_0(فراغ * معلومات, ثابت ggml_tensor * t, size_t دننه_واټن, size_t size) {
    unpack_from_tiled_grid_32<بلاک_q4_0, 1>(معلومات, t, دننه_واټن, size, HTP_TENSOR_TILE_اندازه_Q4_0,
        [](بلاک_q4_0 * dst_ټوټه, ثابت uint8_t * tile_src, int64_t ne0, int64_t پیل_لیکه, int64_t پای_ټکی_لیکه, int64_t لیکه_stride, صحيح ct, صحيح kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_منزل_nibbles(tile_quants, tile_src);
            ثابت ggml_half * scale_src = (ثابت ggml_half *) (tile_src + 512);
            لكل (صحيح لیکه = 0; لیکه < 32; لیکه++) {
                ثابت int64_t r = ct * 32 + لیکه;
                إذا (r >= پیل_لیکه && r < پای_ټکی_لیکه && kt < ne0 / 32) {
                    بلاک_q4_0 & b = dst_ټوټه[(r - پیل_لیکه) * لیکه_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[لیکه]);
                    b.d = scale_src[لیکه];
                }
            }
        });
}

ساكن فراغ repack_q4_1_tiled(ggml_tensor * t, ثابت فراغ * معلومات, size_t دننه_واټن, size_t size) {
    repack_منزل_tiled_grid_32<بلاک_q4_1, 1>(t, معلومات, دننه_واټن, size, HTP_TENSOR_TILE_اندازه_Q4_1,
        [](uint8_t * tile_dst, ثابت بلاک_q4_1 * src_ټوټه, int64_t ne0, int64_t ne1, int64_t لیکه_stride, صحيح ct, صحيح kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
#إذا defined(__ARM_NEON) || defined(__aarch64__)
            إذا (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                لكل (صحيح row0 = 0; row0 < 32; row0 += 8) {
                    ثابت uint8_t * qs[8];
                    لكل (صحيح rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_ټوټه[(ct * 32 + row0 + rr) * لیکه_stride + kt].qs;
                    }
                    repack_neon_4bit_لیکې(qs, tile_dst, row0);
                }

                لكل (صحيح لیکه = 0; لیکه < 32; ++لیکه) {
                    ثابت بلاک_q4_1 & src_بلاک = src_ټوټه[(ct * 32 + لیکه) * لیکه_stride + kt];
                    scale_dst[2 * لیکه + 0] = src_بلاک.d;
                    scale_dst[2 * لیکه + 1] = src_بلاک.m;
                }
                أرجع;
            }
#د_که_پای
            uint8_t tile_quants[32][32];
            لكل (صحيح لیکه = 0; لیکه < 32; لیکه++) {
                ثابت int64_t r = ct * 32 + لیکه;
                إذا (r < ne1 && kt < ne0 / 32) {
                    ثابت بلاک_q4_1 & b = src_ټوټه[r * لیکه_stride + kt];
                    unpack_nibbles_32(tile_quants[لیکه], b.qs);
                    scale_dst[2 * لیکه + 0] = b.d;
                    scale_dst[2 * لیکه + 1] = b.m;
                } وإلا {
                    memset(tile_quants[لیکه], 0, 32);
                    scale_dst[2 * لیکه + 0] = 0;
                    scale_dst[2 * لیکه + 1] = 0;
                }
            }
            transpose_nibbles_منزل_tile(tile_dst, tile_quants);
        });
}

ساكن فراغ repack_tiled_q4_1(فراغ * معلومات, ثابت ggml_tensor * t, size_t دننه_واټن, size_t size) {
    unpack_from_tiled_grid_32<بلاک_q4_1, 1>(معلومات, t, دننه_واټن, size, HTP_TENSOR_TILE_اندازه_Q4_1,
        [](بلاک_q4_1 * dst_ټوټه, ثابت uint8_t * tile_src, int64_t ne0, int64_t پیل_لیکه, int64_t پای_ټکی_لیکه, int64_t لیکه_stride, صحيح ct, صحيح kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_منزل_nibbles(tile_quants, tile_src);
            ثابت ggml_half * scale_src = (ثابت ggml_half *) (tile_src + 512);
            لكل (صحيح لیکه = 0; لیکه < 32; لیکه++) {
                ثابت int64_t r = ct * 32 + لیکه;
                إذا (r >= پیل_لیکه && r < پای_ټکی_لیکه && kt < ne0 / 32) {
                    بلاک_q4_1 & b = dst_ټوټه[(r - پیل_لیکه) * لیکه_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[لیکه]);
                    b.d = scale_src[2 * لیکه + 0];
                    b.m = scale_src[2 * لیکه + 1];
                }
            }
        });
}

ساكن فراغ repack_mxfp4_tiled(ggml_tensor * t, ثابت فراغ * معلومات, size_t دننه_واټن, size_t size) {
    repack_منزل_tiled_grid_32<بلاک_mxfp4, 1>(t, معلومات, دننه_واټن, size, HTP_TENSOR_TILE_اندازه_MXFP4,
        [](uint8_t * tile_dst, ثابت بلاک_mxfp4 * src_ټوټه, int64_t ne0, int64_t ne1, int64_t لیکه_stride, صحيح ct, صحيح kt) {
            uint8_t * scale_dst = tile_dst + 512;
#إذا defined(__ARM_NEON) || defined(__aarch64__)
            إذا (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                لكل (صحيح row0 = 0; row0 < 32; row0 += 8) {
                    ثابت uint8_t * qs[8];
                    لكل (صحيح rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_ټوټه[(ct * 32 + row0 + rr) * لیکه_stride + kt].qs;
                    }
                    repack_neon_4bit_لیکې(qs, tile_dst, row0);
                }

                لكل (صحيح لیکه = 0; لیکه < 32; ++لیکه) {
                    scale_dst[لیکه] = src_ټوټه[(ct * 32 + لیکه) * لیکه_stride + kt].e;
                }
                أرجع;
            }
#د_که_پای
            uint8_t tile_quants[32][32];
            لكل (صحيح لیکه = 0; لیکه < 32; لیکه++) {
                ثابت int64_t r = ct * 32 + لیکه;
                إذا (r < ne1 && kt < ne0 / 32) {
                    ثابت بلاک_mxfp4 & b = src_ټوټه[r * لیکه_stride + kt];
                    unpack_nibbles_32(tile_quants[لیکه], b.qs);
                    scale_dst[لیکه] = b.e;
                } وإلا {
                    memset(tile_quants[لیکه], 0, 32);
                    scale_dst[لیکه] = 0;
                }
            }
            transpose_nibbles_منزل_tile(tile_dst, tile_quants);
        });
}

ساكن فراغ repack_tiled_mxfp4(فراغ * معلومات, ثابت ggml_tensor * t, size_t دننه_واټن, size_t size) {
    unpack_from_tiled_grid_32<بلاک_mxfp4, 1>(معلومات, t, دننه_واټن, size, HTP_TENSOR_TILE_اندازه_MXFP4,
        [](بلاک_mxfp4 * dst_ټوټه, ثابت uint8_t * tile_src, int64_t ne0, int64_t پیل_لیکه, int64_t پای_ټکی_لیکه, int64_t لیکه_stride, صحيح ct, صحيح kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_منزل_nibbles(tile_quants, tile_src);
            ثابت uint8_t * scale_src = tile_src + 512;
            لكل (صحيح لیکه = 0; لیکه < 32; لیکه++) {
                ثابت int64_t r = ct * 32 + لیکه;
                إذا (r >= پیل_لیکه && r < پای_ټکی_لیکه && kt < ne0 / 32) {
                    بلاک_mxfp4 & b = dst_ټوټه[(r - پیل_لیکه) * لیکه_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[لیکه]);
                    b.e = scale_src[لیکه];
                }
            }
        });
}

ساكن فراغ repack_q8_0_tiled(ggml_tensor * t, ثابت فراغ * معلومات, size_t دننه_واټن, size_t size) {
    repack_منزل_tiled_grid_32<بلاک_q8_0, 1>(t, معلومات, دننه_واټن, size, HTP_TENSOR_TILE_اندازه_Q8_0,
        [](uint8_t * tile_dst, ثابت بلاک_q8_0 * src_ټوټه, int64_t ne0, int64_t ne1, int64_t لیکه_stride, صحيح ct, صحيح kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 1024);
#إذا defined(__ARM_NEON) || defined(__aarch64__)
            إذا (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                لكل (صحيح row0 = 0; row0 < 32; row0 += 8) {
                    ثابت uint8_t * qs[8];
                    لكل (صحيح rr = 0; rr < 8; ++rr) {
                        qs[rr] = (ثابت uint8_t *) src_ټوټه[(ct * 32 + row0 + rr) * لیکه_stride + kt].qs;
                    }
                    repack_neon_q8_لیکې(qs, tile_dst, row0);
                }

                لكل (صحيح لیکه = 0; لیکه < 32; ++لیکه) {
                    scale_dst[لیکه] = src_ټوټه[(ct * 32 + لیکه) * لیکه_stride + kt].d;
                }
                أرجع;
            }
#د_که_پای
            لكل (صحيح cp = 0; cp < 16; cp++) {
                ثابت صحيح col0 = cp * 2;
                ثابت صحيح col1 = col0 + 1;
                لكل (صحيح لیکه = 0; لیکه < 32; لیکه++) {
                    ثابت int64_t r = ct * 32 + لیکه;
                    ثابت بلاک_q8_0 * b = (r < ne1 && kt < ne0 / 32) ? &src_ټوټه[r * لیکه_stride + kt] : تش_اشاره;
                    tile_dst[cp * 64 + 2 * لیکه + 0] = b ? b->qs[col0] : 0;
                    tile_dst[cp * 64 + 2 * لیکه + 1] = b ? b->qs[col1] : 0;
                }
            }
            لكل (صحيح لیکه = 0; لیکه < 32; لیکه++) {
                ثابت int64_t r = ct * 32 + لیکه;
                scale_dst[لیکه] = (r < ne1 && kt < ne0 / 32) ? src_ټوټه[r * لیکه_stride + kt].d : 0;
            }
        });
}

ساكن فراغ repack_tiled_q8_0(فراغ * معلومات, ثابت ggml_tensor * t, size_t دننه_واټن, size_t size) {
    unpack_from_tiled_grid_32<بلاک_q8_0, 1>(معلومات, t, دننه_واټن, size, HTP_TENSOR_TILE_اندازه_Q8_0,
        [](بلاک_q8_0 * dst_ټوټه, ثابت uint8_t * tile_src, int64_t ne0, int64_t پیل_لیکه, int64_t پای_ټکی_لیکه, int64_t لیکه_stride, صحيح ct, صحيح kt) {
            لكل (صحيح cp = 0; cp < 16; cp++) {
                ثابت صحيح col0 = cp * 2;
                ثابت صحيح col1 = col0 + 1;
                لكل (صحيح لیکه = 0; لیکه < 32; لیکه++) {
                    ثابت int64_t r = ct * 32 + لیکه;
                    إذا (r >= پیل_لیکه && r < پای_ټکی_لیکه && kt < ne0 / 32) {
                        بلاک_q8_0 & b = dst_ټوټه[(r - پیل_لیکه) * لیکه_stride + kt];
                        b.qs[col0] = tile_src[cp * 64 + 2 * لیکه + 0];
                        b.qs[col1] = tile_src[cp * 64 + 2 * لیکه + 1];
                    }
                }
            }
            ثابت ggml_half * scale_src = (ثابت ggml_half *) (tile_src + 1024);
            لكل (صحيح لیکه = 0; لیکه < 32; لیکه++) {
                ثابت int64_t r = ct * 32 + لیکه;
                إذا (r >= پیل_لیکه && r < پای_ټکی_لیکه && kt < ne0 / 32) {
                    dst_ټوټه[(r - پیل_لیکه) * لیکه_stride + kt].d = scale_src[لیکه];
                }
            }
        });
}

// --- Float Types (F16, F32, BF16 -> f16_tiled HMX layout) ---

ساكن فراغ repack_q2_0_tiled(ggml_tensor * t, ثابت فراغ * معلومات, size_t دننه_واټن, size_t size) {
    GGML_ASSERT(دننه_واټن == 0);
    GGML_ASSERT(t->ne[0] % 64 == 0);
    GGML_ASSERT(size == ggml_nbytes(t));

    ثابت بلاک_q2_0 * src = (ثابت بلاک_q2_0 *) معلومات;
    ثابت int64_t بلاکونه_فی_لیکه = t->ne[0] / 64;
    ثابت int64_t n_ستون_tiles = hex_پړاو_پورته(t->ne[1], 32) / 32;
    ثابت int64_t n_وزن_tiles = بلاکونه_فی_لیکه;
    ثابت size_t ماتریکس_اندازه = (size_t) n_ستون_tiles * n_وزن_tiles * HTP_TENSOR_TILE_اندازه_Q2_0;

    لكل (int64_t ټوټه_2 = 0; ټوټه_2 < t->ne[2] * t->ne[3]; ++ټوټه_2) {
        ثابت بلاک_q2_0 * src_ټوټه = src + ټوټه_2 * t->ne[1] * بلاکونه_فی_لیکه;
        uint8_t * dst_ټوټه = (uint8_t *) t->معلومات + ټوټه_2 * ماتریکس_اندازه;
        ثابت size_t ct_بایټونه = (size_t) n_وزن_tiles * HTP_TENSOR_TILE_اندازه_Q2_0;

        repack_parallel_for_ct(ماتریکس_اندازه, (صحيح) n_ستون_tiles, [&](صحيح ct0, صحيح ct1) {
            memset(dst_ټوټه + (size_t) ct0 * ct_بایټونه, 0, (size_t) (ct1 - ct0) * ct_بایټونه);
            ثابت int64_t r0 = (int64_t) ct0 * 32;
            ثابت int64_t r1 = (std::min)((int64_t) ct1 * 32, t->ne[1]);
            لكل (int64_t لیکه_شاخص = r0; لیکه_شاخص < r1; ++لیکه_شاخص) {
                ثابت int64_t ct = لیکه_شاخص / 32;
                ثابت صحيح لیکه = (صحيح) (لیکه_شاخص % 32);
                لكل (int64_t kt = 0; kt < n_وزن_tiles; ++kt) {
                    ثابت بلاک_q2_0 * بلاک = src_ټوټه + لیکه_شاخص * بلاکونه_فی_لیکه + kt;
                    uint8_t * tile = dst_ټوټه + (ct * n_وزن_tiles + kt) * HTP_TENSOR_TILE_اندازه_Q2_0;
#إذا defined(__ARM_NEON) || defined(__aarch64__)
                    repack_neon_q2_0_بلاک(بلاک->qs, tile, لیکه);
#وإلا
                    لكل (صحيح ډله = 0; ډله < 16; ++ډله) {
                        لكل (صحيح byte = 0; byte < 4; ++byte) {
                            ثابت صحيح k = 4 * ډله + byte;
                            ثابت uint8_t code = (بلاک->qs[k / 4] >> (2 * (k % 4))) & 3;
                            tile[(ډله >> 2) * 128 + 4 * لیکه + byte] |= code << (2 * (ډله & 3));
                        }
                    }
#د_که_پای
                    حافظه_نقل(tile + 512 + 2 * لیکه, &بلاک->d, حجم(بلاک->d));
                }
            }
        });
    }
}

ساكن فراغ repack_tiled_q2_0(فراغ * معلومات, ثابت ggml_tensor * t, size_t دننه_واټن, size_t size) {
    GGML_ASSERT(t->ne[0] % 64 == 0);
    GGML_ASSERT(دننه_واټن + size <= ggml_nbytes(t));

    ثابت size_t لومړی_بلاک = دننه_واټن / حجم(بلاک_q2_0);
    ثابت size_t پای_ټکی_بلاک = (دننه_واټن + size + حجم(بلاک_q2_0) - 1) / حجم(بلاک_q2_0);
    ثابت size_t بلاکونه_فی_لیکه = t->ne[0] / 64;
    ثابت size_t بلاکونه_فی_ټوټه = t->ne[1] * بلاکونه_فی_لیکه;
    ثابت size_t n_وزن_tiles = بلاکونه_فی_لیکه;
    ثابت size_t n_ستون_tiles = hex_پړاو_پورته(t->ne[1], 32) / 32;
    ثابت size_t ماتریکس_اندازه = n_ستون_tiles * n_وزن_tiles * HTP_TENSOR_TILE_اندازه_Q2_0;

    لكل (size_t بلاک_شاخص = لومړی_بلاک; بلاک_شاخص < پای_ټکی_بلاک; ++بلاک_شاخص) {
        ثابت size_t ټوټه_2 = بلاک_شاخص / بلاکونه_فی_ټوټه;
        ثابت size_t دننه_ټوټه = بلاک_شاخص % بلاکونه_فی_ټوټه;
        ثابت size_t لیکه_شاخص = دننه_ټوټه / بلاکونه_فی_لیکه;
        ثابت size_t kt = دننه_ټوټه % بلاکونه_فی_لیکه;
        ثابت size_t ct = لیکه_شاخص / 32;
        ثابت size_t لیکه = لیکه_شاخص % 32;
        ثابت uint8_t * tile = (ثابت uint8_t *) t->معلومات + ټوټه_2 * ماتریکس_اندازه +
            (ct * n_وزن_tiles + kt) * HTP_TENSOR_TILE_اندازه_Q2_0;

        بلاک_q2_0 بلاک = {};
        حافظه_نقل(&بلاک.d, tile + 512 + 2 * لیکه, حجم(بلاک.d));
        لكل (صحيح ډله = 0; ډله < 16; ++ډله) {
            لكل (صحيح byte = 0; byte < 4; ++byte) {
                ثابت uint8_t packed = tile[(ډله >> 2) * 128 + 4 * لیکه + byte];
                ثابت uint8_t code = (packed >> (2 * (ډله & 3))) & 3;
                بلاک.qs[ډله] |= code << (2 * byte);
            }
        }

        ثابت size_t بلاک_begin = بلاک_شاخص * حجم(بلاک_q2_0);
        ثابت size_t copy_begin = (std::max)(بلاک_begin, دننه_واټن);
        ثابت size_t copy_پای_ټکی = (std::min)(بلاک_begin + حجم(بلاک_q2_0), دننه_واټن + size);
        حافظه_نقل((uint8_t *) معلومات + copy_begin - دننه_واټن, (ثابت uint8_t *) &بلاک + copy_begin - بلاک_begin, copy_پای_ټکی - copy_begin);
    }
}

کينډۍ <ډول_نوم SrcT, ډول_نوم منزل_Fp16_Fn>
ساكن فراغ repack_float_منزل_f16_tiled(ggml_tensor * t, ثابت فراغ * معلومات, size_t دننه_واټن, size_t size, منزل_Fp16_Fn && منزل_fp16) {
    repack_منزل_tiled_grid_32<SrcT, 32>(t, معلومات, دننه_واټن, size, HTP_TENSOR_TILE_اندازه_F16,
        [&](uint8_t * tile_بایټونه, ثابت SrcT * src_ټوټه, int64_t ne0, int64_t ne1, int64_t لیکه_stride, صحيح ct, صحيح kt) {
            uint16_t * tile_dst = (uint16_t *) tile_بایټونه;
#إذا defined(__ARM_NEON) || defined(__aarch64__)
            إذا (ct * 32 + 32 <= ne1 && kt * 32 + 32 <= ne0) {
                repack_neon_f16_tile(tile_dst, src_ټوټه, لیکه_stride, ct, kt);
                أرجع;
            }
#د_که_پای
            لكل (صحيح cp = 0; cp < 16; cp++) {
                ثابت int64_t c0 = kt * 32 + cp * 2;
                ثابت int64_t c1 = c0 + 1;
                لكل (صحيح لیکه = 0; لیکه < 32; لیکه++) {
                    ثابت int64_t r = ct * 32 + لیکه;
                    ثابت SrcT * r_ptr = (r < ne1) ? &src_ټوټه[r * لیکه_stride] : تش_اشاره;
                    tile_dst[cp * 64 + لیکه * 2 + 0] = (r_ptr && c0 < ne0) ? منزل_fp16(r_ptr[c0]) : 0;
                    tile_dst[cp * 64 + لیکه * 2 + 1] = (r_ptr && c1 < ne0) ? منزل_fp16(r_ptr[c1]) : 0;
                }
            }
        });
}

کينډۍ <ډول_نوم DstT, ډول_نوم FromFp16Fn>
ساكن فراغ unpack_f16_tiled_منزل_float(فراغ * معلومات, ثابت ggml_tensor * t, size_t دننه_واټن, size_t size, FromFp16Fn && from_fp16) {
    unpack_from_tiled_grid_32<DstT, 32>(معلومات, t, دننه_واټن, size, HTP_TENSOR_TILE_اندازه_F16,
        [&](DstT * dst_ټوټه, ثابت uint8_t * tile_بایټونه, int64_t ne0, int64_t پیل_لیکه, int64_t پای_ټکی_لیکه, int64_t لیکه_stride, صحيح ct, صحيح kt) {
            ثابت uint16_t * tile_src = (ثابت uint16_t *) tile_بایټونه;
            لكل (صحيح cp = 0; cp < 16; cp++) {
                ثابت int64_t c0 = kt * 32 + cp * 2;
                ثابت int64_t c1 = c0 + 1;
                لكل (صحيح لیکه = 0; لیکه < 32; لیکه++) {
                    ثابت int64_t r = ct * 32 + لیکه;
                    إذا (r >= پیل_لیکه && r < پای_ټکی_لیکه) {
                        DstT * r_ptr = dst_ټوټه + (r - پیل_لیکه) * لیکه_stride;
                        إذا (c0 < ne0) r_ptr[c0] = from_fp16(tile_src[cp * 64 + لیکه * 2 + 0]);
                        إذا (c1 < ne0) r_ptr[c1] = from_fp16(tile_src[cp * 64 + لیکه * 2 + 1]);
                    }
                }
            }
        });
}

// --- Generic K-Quant (super-block 256) grid walkers ---

کينډۍ <ډول_نوم بلاک_T, ډول_نوم PackKtFn>
ساكن فراغ repack_kquant_منزل_tiled_grid(ggml_tensor * t, ثابت فراغ * معلومات, size_t دننه_واټن, size_t size,
                                        size_t tile_اندازه, PackKtFn && pack_kt) {
    GGML_ASSERT(دننه_واټن == 0);
    GGML_ناکارول_شوی(size);

    ثابت بلاک_T * src_ماتریکس = (ثابت بلاک_T *) معلومات;
    ثابت int64_t ne0 = t->ne[0];
    ثابت int64_t ne1 = t->ne[1];
    ثابت int64_t ne2 = t->ne[2];
    ثابت int64_t ne3 = t->ne[3];
    GGML_ASSERT(ne0 % QK_K == 0);

    ثابت صحيح n_ستون_tiles    = hex_پړاو_پورته(ne1, 32) / 32;
    ثابت صحيح n_k_tiles      = hex_پړاو_پورته(ne0, 32) / 32;
    ثابت size_t ماتریکس_اندازه = (size_t) n_ستون_tiles * n_k_tiles * tile_اندازه;
    ثابت int64_t sb_فی_لیکه = ne0 / QK_K;

    لكل (int64_t ټوټه_idx = 0; ټوټه_idx < ne2 * ne3; ټوټه_idx++) {
        ثابت بلاک_T * src_ټوټه = src_ماتریکس + ټوټه_idx * (ne1 * sb_فی_لیکه);
        uint8_t * ماتریکس_dst     = (uint8_t *) t->معلومات + ټوټه_idx * ماتریکس_اندازه;
        ثابت size_t ct_بایټونه    = (size_t) n_k_tiles * tile_اندازه;

        repack_parallel_for_ct(ماتریکس_اندازه, n_ستون_tiles, [&](صحيح ct0, صحيح ct1) {
            // Padding rows and OR-ed bit planes require zeroed tiles
            memset(ماتریکس_dst + (size_t) ct0 * ct_بایټونه, 0, (size_t) (ct1 - ct0) * ct_بایټونه);

            ثابت int64_t r0 = (int64_t) ct0 * 32;
            ثابت int64_t r1 = (std::min)((int64_t) ct1 * 32, ne1);
            لكل (int64_t r = r0; r < r1; r++) {
                ثابت صحيح ct  = (صحيح) (r / 32);
                ثابت صحيح لیکه = (صحيح) (r % 32);
                ثابت بلاک_T * src_لیکه = src_ټوټه + r * sb_فی_لیکه;

                لكل (صحيح kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile = ماتریکس_dst + ((size_t) ct * n_k_tiles + kt) * tile_اندازه;
                    pack_kt(tile, &src_لیکه[kt / 8], لیکه, kt % 8);
                }
            }
        });
    }
}

کينډۍ <ډول_نوم بلاک_T, ډول_نوم UnpackSbFn>
ساكن فراغ unpack_kquant_from_tiled_grid(فراغ * معلومات, ثابت ggml_tensor * t, size_t دننه_واټن, size_t size,
                                          size_t tile_اندازه, UnpackSbFn && unpack_sb) {
    uint8_t * dst_بایټونه = (uint8_t *) معلومات;
    ثابت int64_t ne0 = t->ne[0];
    ثابت int64_t ne1 = t->ne[1];
    ثابت int64_t ne2 = t->ne[2];
    ثابت int64_t ne3 = t->ne[3];
    GGML_ASSERT(ne0 % QK_K == 0);

    ثابت صحيح n_ستون_tiles    = hex_پړاو_پورته(ne1, 32) / 32;
    ثابت صحيح n_k_tiles      = hex_پړاو_پورته(ne0, 32) / 32;
    ثابت size_t ماتریکس_اندازه = (size_t) n_ستون_tiles * n_k_tiles * tile_اندازه;
    ثابت int64_t sb_فی_لیکه = ne0 / QK_K;

    ثابت size_t لیکه_اندازه_بایټونه = ggml_لیکه_اندازه(t->type, ne0);
    ثابت size_t ټوټه_اندازه     = (size_t) ne1 * لیکه_اندازه_بایټونه;
    ثابت int64_t پیل_ټوټه   = دننه_واټن / ټوټه_اندازه;
    int64_t پای_ټکی_ټوټه           = (دننه_واټن + size + ټوټه_اندازه - 1) / ټوټه_اندازه;
    إذا (پای_ټکی_ټوټه > ne2 * ne3) {
        پای_ټکی_ټوټه = ne2 * ne3;
    }

    لكل (int64_t ټوټه_idx = پیل_ټوټه; ټوټه_idx < پای_ټکی_ټوټه; ټوټه_idx++) {
        ثابت size_t cur_پیل_byte     = (std::max)(دننه_واټن, (size_t) ټوټه_idx * ټوټه_اندازه);
        ثابت size_t cur_پای_ټکی_byte       = (std::min)(دننه_واټن + size, (size_t) (ټوټه_idx + 1) * ټوټه_اندازه);
        ثابت size_t ټوټه_دننه_واټن_پیل = cur_پیل_byte - (size_t) ټوټه_idx * ټوټه_اندازه;
        ثابت size_t ټوټه_دننه_واټن_پای_ټکی   = cur_پای_ټکی_byte - (size_t) ټوټه_idx * ټوټه_اندازه;

        ثابت int64_t پیل_لیکه = ټوټه_دننه_واټن_پیل / لیکه_اندازه_بایټونه;
        ثابت int64_t پای_ټکی_لیکه   = (std::min)((int64_t) ((ټوټه_دننه_واټن_پای_ټکی + لیکه_اندازه_بایټونه - 1) / لیکه_اندازه_بایټونه), ne1);

        بلاک_T * dst_ټوټه         = (بلاک_T *) (dst_بایټونه + (cur_پیل_byte - دننه_واټن));
        ثابت uint8_t * ماتریکس_src = (ثابت uint8_t *) t->معلومات + ټوټه_idx * ماتریکس_اندازه;

        لكل (int64_t r = پیل_لیکه; r < پای_ټکی_لیکه; r++) {
            ثابت صحيح ct  = (صحيح) (r / 32);
            ثابت صحيح لیکه = (صحيح) (r % 32);
            بلاک_T * dst_لیکه = dst_ټوټه + (r - پیل_لیکه) * sb_فی_لیکه;

            لكل (int64_t sb = 0; sb < sb_فی_لیکه; sb++) {
                بلاک_T * b = &dst_لیکه[sb];
                memset(b, 0, حجم(بلاک_T));
                ثابت uint8_t * sb_tiles = ماتریکس_src + ((size_t) ct * n_k_tiles + sb * 8) * tile_اندازه;
                unpack_sb(b, sb_tiles, tile_اندازه, لیکه);
            }
        }
    }
}

// --- K-Quant Helpers & Per-Type Implementations (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) ---

#إذا !defined(__ARM_NEON) && !defined(__aarch64__)
// unsigned 6-bit value (0..63) of element e of a Q6_K block, same bit layout as dequantize_row_q6_K
ساكن مضمن uint8_t q6_K_get_quant(ثابت بلاک_q6_K * b, صحيح e) {
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
#د_که_پای

ساكن فراغ repack_q6_K_tiled(ggml_tensor * t, ثابت فراغ * معلومات, size_t دننه_واټن, size_t size) {
    repack_kquant_منزل_tiled_grid<بلاک_q6_K>(t, معلومات, دننه_واټن, size, HTP_TENSOR_TILE_اندازه_Q6_K,
        [](uint8_t * tile, ثابت بلاک_q6_K * b, صحيح لیکه, صحيح kt_local) {
            ثابت عائم d     = GGML_FP16_منزل_FP32(b->d);
            uint8_t * lo_pl   = tile;
            uint8_t * hi_pl   = tile + 512;
            ggml_half * sc_pl = (ggml_half *) (tile + 768);

#إذا defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q6_k(b, kt_local, lo_pl, hi_pl, لیکه);
#وإلا
            لكل (صحيح lk = 0; lk < 32; lk++) {
                ثابت uint8_t q6 = q6_K_get_quant(b, kt_local * 32 + lk);
                ثابت صحيح g   = lk >> 2;
                ثابت صحيح pos = لیکه * 4 + (lk & 3);
                lo_pl[(g >> 1) * 128 + pos] |= (uint8_t) ((q6 & 0xF) << ((g & 1) * 4));
                hi_pl[(g >> 2) * 128 + pos] |= (uint8_t) ((q6 >> 4) << ((g & 3) * 2));
            }
#د_که_پای
            لكل (صحيح sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + لیکه] = GGML_FP32_منزل_FP16(d * (عائم) b->scales[kt_local * 2 + sub]);
            }
        });
}

ساكن فراغ repack_tiled_q6_K(فراغ * معلومات, ثابت ggml_tensor * t, size_t دننه_واټن, size_t size) {
    unpack_kquant_from_tiled_grid<بلاک_q6_K>(معلومات, t, دننه_واټن, size, HTP_TENSOR_TILE_اندازه_Q6_K,
        [](بلاک_q6_K * b, ثابت uint8_t * sb_tiles, size_t tile_اندازه, صحيح لیکه) {
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
                    ثابت صحيح pos    = لیکه * 4 + (lk & 3);
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
                    sub_scales[kt_local * 2 + sub] = GGML_FP16_منزل_FP32(sc_pl[sub * 32 + لیکه]);
                }
            }

            عائم max_abs_scale = 0.0f;
            لكل (صحيح s = 0; s < 16; s++) {
                max_abs_scale = (std::max)(max_abs_scale, fabsf(sub_scales[s]));
            }

            إذا (max_abs_scale == 0.0f) {
                b->d = GGML_FP32_منزل_FP16(0.0f);
            } وإلا {
                b->d = GGML_FP32_منزل_FP16(max_abs_scale / 127.0f);
                ثابت عائم d_actual = GGML_FP16_منزل_FP32(b->d);
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
ساكن مضمن منطقي q3_K_get_hbit(ثابت بلاک_q3_K * b, صحيح e) {
    ثابت صحيح c = e / 128;
    ثابت صحيح j = (e % 128) / 32;
    ثابت صحيح l = e % 32;
    أرجع (b->hmask[l] >> (c * 4 + j)) & 1;
}
#د_که_پای

// signed 6-bit scale j (-32..31) of a Q3_K block, same packing as quantize_row_q3_K_ref
ساكن مضمن صحيح q3_K_get_scale(ثابت uint8_t * scales, صحيح j) {
    ثابت صحيح lo = (j < 8) ? (scales[j] & 0xF) : (scales[j - 8] >> 4);
    ثابت صحيح hi = (scales[8 + j % 4] >> (2 * (j / 4))) & 3;
    أرجع (lo | (hi << 4)) - 32;
}

// read-back: find fp16 d and l[j] in [lmin, lmax] with fp16(d * l[j]) == prod[j] for all j, false if none
ساكن منطقي hexagon_recover_k_scales(ثابت ggml_half * prod, صحيح n, صحيح lmin, صحيح lmax, ggml_half * d_بهر, صحيح * l_بهر) {
    صحيح jmax = 0;
    لكل (صحيح j = 1; j < n; j++) {
        إذا (fabsf(GGML_FP16_منزل_FP32(prod[j])) > fabsf(GGML_FP16_منزل_FP32(prod[jmax]))) {
            jmax = j;
        }
    }
    ثابت عائم pmax = GGML_FP16_منزل_FP32(prod[jmax]);
    إذا (pmax == 0.0f) {
        *d_بهر = GGML_FP32_منزل_FP16(0.0f);
        لكل (صحيح j = 0; j < n; j++) {
            l_بهر[j] = 0;
        }
        أرجع ریښتیا;
    }

    // the quantizers put the largest scale at or near the range end, so try large |l| first
    ثابت صحيح lext = (std::max)(-lmin, lmax);
    لكل (صحيح a = lext; a >= 1; a--) {
        لكل (صحيح sign : { 1, -1 }) {
            ثابت صحيح lj = sign * a;
            إذا (lj < lmin || lj > lmax) {
                استمر;
            }
            ثابت ggml_half d0 = GGML_FP32_منزل_FP16(pmax / (عائم) lj);
            لكل (صحيح ulp : { 0, -1, 1 }) {
                ggml_half d = d0;
                uint16_t بټونه;
                حافظه_نقل(&بټونه, &d, حجم(بټونه));
                بټونه = (uint16_t) (بټونه + ulp);
                حافظه_نقل(&d, &بټونه, حجم(بټونه));

                ثابت عائم df = GGML_FP16_منزل_FP32(d);
                إذا (!std::isfinite(df) || df == 0.0f) {
                    استمر;
                }
                منطقي ok = ریښتیا;
                لكل (صحيح j = 0; j < n && ok; j++) {
                    ثابت صحيح l = (صحيح) roundf(GGML_FP16_منزل_FP32(prod[j]) / df);
                    ثابت ggml_half p = GGML_FP32_منزل_FP16(df * (عائم) l);
                    ok = l >= lmin && l <= lmax && memcmp(&p, &prod[j], حجم(p)) == 0;
                    l_بهر[j] = l;
                }
                إذا (ok) {
                    *d_بهر = d;
                    أرجع ریښتیا;
                }
            }
        }
    }
    أرجع خطأ;
}

ساكن فراغ repack_q3_K_tiled(ggml_tensor * t, ثابت فراغ * معلومات, size_t دننه_واټن, size_t size) {
    repack_kquant_منزل_tiled_grid<بلاک_q3_K>(t, معلومات, دننه_واټن, size, HTP_TENSOR_TILE_اندازه_Q3_K,
        [](uint8_t * tile, ثابت بلاک_q3_K * b, صحيح لیکه, صحيح kt_local) {
            ثابت عائم d     = GGML_FP16_منزل_FP32(b->d);
            uint8_t * lo_pl   = tile;
            uint8_t * neg_pl  = tile + 256;
            ggml_half * sc_pl = (ggml_half *) (tile + 384);

#إذا defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q2_3_k_low2(b->qs, kt_local, lo_pl, لیکه);
            repack_neon_q3_k_hmask(b->hmask, kt_local, neg_pl, لیکه);
#وإلا
            لكل (صحيح lk = 0; lk < 32; lk++) {
                ثابت صحيح e   = kt_local * 32 + lk;
                ثابت صحيح g   = lk >> 2;
                ثابت صحيح pos = لیکه * 4 + (lk & 3);
                lo_pl[(g >> 2) * 128 + pos] |= (uint8_t) (q2_3_K_get_low2(b->qs, e) << ((g & 3) * 2));
                إذا (!q3_K_get_hbit(b, e)) {
                    neg_pl[pos] |= (uint8_t) (1 << g);
                }
            }
#د_که_پای
            لكل (صحيح sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + لیکه] = GGML_FP32_منزل_FP16(d * (عائم) q3_K_get_scale(b->scales, kt_local * 2 + sub));
            }
        });
}

ساكن فراغ repack_tiled_q3_K(فراغ * معلومات, ثابت ggml_tensor * t, size_t دننه_واټن, size_t size) {
    unpack_kquant_from_tiled_grid<بلاک_q3_K>(معلومات, t, دننه_واټن, size, HTP_TENSOR_TILE_اندازه_Q3_K,
        [](بلاک_q3_K * b, ثابت uint8_t * sb_tiles, size_t tile_اندازه, صحيح لیکه) {
            ggml_half sub_scales[16];
            لكل (صحيح kt_local = 0; kt_local < 8; kt_local++) {
                ثابت uint8_t *   tile   = sb_tiles + kt_local * tile_اندازه;
                ثابت uint8_t *   lo_pl  = tile;
                ثابت uint8_t *   neg_pl = tile + 256;
                ثابت ggml_half * sc_pl  = (ثابت ggml_half *) (tile + 384);

                لكل (صحيح lk = 0; lk < 32; lk++) {
                    ثابت صحيح e   = kt_local * 32 + lk;
                    ثابت صحيح g   = lk >> 2;
                    ثابت صحيح pos = لیکه * 4 + (lk & 3);
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
                    sub_scales[kt_local * 2 + sub] = sc_pl[sub * 32 + لیکه];
                }
            }

            صحيح ls[16];
            إذا (!hexagon_recover_k_scales(sub_scales, 16, -32, 31, &b->d, ls)) {
                عائم max_scale = 0.0f;
                لكل (صحيح s = 0; s < 16; s++) {
                    إذا (fabsf(GGML_FP16_منزل_FP32(sub_scales[s])) > fabsf(max_scale)) {
                        max_scale = GGML_FP16_منزل_FP32(sub_scales[s]);
                    }
                }
                b->d = GGML_FP32_منزل_FP16(-max_scale / 32.0f);
                ثابت عائم d_actual = GGML_FP16_منزل_FP32(b->d);
                ثابت عائم inv_d    = (d_actual != 0.0f) ? (1.0f / d_actual) : 0.0f;
                لكل (صحيح s = 0; s < 16; s++) {
                    ls[s] = (std::max)(-32, (std::min)(31, (صحيح) roundf(GGML_FP16_منزل_FP32(sub_scales[s]) * inv_d)));
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

ساكن فراغ repack_q2_K_tiled(ggml_tensor * t, ثابت فراغ * معلومات, size_t دننه_واټن, size_t size) {
    repack_kquant_منزل_tiled_grid<بلاک_q2_K>(t, معلومات, دننه_واټن, size, HTP_TENSOR_TILE_اندازه_Q2_K,
        [](uint8_t * tile, ثابت بلاک_q2_K * b, صحيح لیکه, صحيح kt_local) {
            uint8_t * lo_pl   = tile;
            uint8_t * sc_pl   = tile + 256;
            ggml_half * dd_pl = (ggml_half *) (tile + 320);

#إذا defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q2_3_k_low2(b->qs, kt_local, lo_pl, لیکه);
#وإلا
            لكل (صحيح lk = 0; lk < 32; lk++) {
                ثابت صحيح g   = lk >> 2;
                ثابت صحيح pos = لیکه * 4 + (lk & 3);
                lo_pl[(g >> 2) * 128 + pos] |= (uint8_t) (q2_3_K_get_low2(b->qs, kt_local * 32 + lk) << ((g & 3) * 2));
            }
#د_که_پای
            لكل (صحيح sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + لیکه] = b->scales[kt_local * 2 + sub];
            }
            إذا (لیکه / 8 == kt_local % 4) {
                dd_pl[لیکه % 8] = kt_local < 4 ? b->d : (ggml_half) (b->dmin ^ 0x8000);
            }
        });
}

ساكن فراغ repack_tiled_q2_K(فراغ * معلومات, ثابت ggml_tensor * t, size_t دننه_واټن, size_t size) {
    unpack_kquant_from_tiled_grid<بلاک_q2_K>(معلومات, t, دننه_واټن, size, HTP_TENSOR_TILE_اندازه_Q2_K,
        [](بلاک_q2_K * b, ثابت uint8_t * sb_tiles, size_t tile_اندازه, صحيح لیکه) {
            لكل (صحيح kt_local = 0; kt_local < 8; kt_local++) {
                ثابت uint8_t *   tile  = sb_tiles + kt_local * tile_اندازه;
                ثابت uint8_t *   lo_pl = tile;
                ثابت uint8_t *   sc_pl = tile + 256;
                ثابت ggml_half * dd_pl = (ثابت ggml_half *) (tile + 320);

                لكل (صحيح lk = 0; lk < 32; lk++) {
                    ثابت صحيح e   = kt_local * 32 + lk;
                    ثابت صحيح g   = lk >> 2;
                    ثابت صحيح pos = لیکه * 4 + (lk & 3);
                    ثابت uint8_t lo = (lo_pl[(g >> 2) * 128 + pos] >> ((g & 3) * 2)) & 3;
                    b->qs[(e / 128) * 32 + e % 32] |= (uint8_t) (lo << (2 * ((e % 128) / 32)));
                }

                لكل (صحيح sub = 0; sub < 2; sub++) {
                    b->scales[kt_local * 2 + sub] = sc_pl[sub * 32 + لیکه];
                }
                إذا (لیکه / 8 == kt_local % 4) {
                    إذا (kt_local < 4) {
                        b->d = dd_pl[لیکه % 8];
                    } وإلا {
                        b->dmin = (ggml_half) (dd_pl[لیکه % 8] ^ 0x8000);
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
        *d = GGML_FP32_منزل_FP16(max_scale / 63.0f);
        ثابت عائم d_actual = GGML_FP16_منزل_FP32(*d);
        inv_scale = (d_actual > 0.0f) ? (1.0f / d_actual) : 0.0f;
    } وإلا {
        *d = GGML_FP32_منزل_FP16(0.0f);
    }

    عائم inv_min = 0.0f;
    إذا (max_min > 0.0f) {
        *dmin = GGML_FP32_منزل_FP16(max_min / 63.0f);
        ثابت عائم dmin_actual = GGML_FP16_منزل_FP32(*dmin);
        inv_min = (dmin_actual > 0.0f) ? (1.0f / dmin_actual) : 0.0f;
    } وإلا {
        *dmin = GGML_FP32_منزل_FP16(0.0f);
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

کينډۍ <ډول_نوم بلاک_T>
ساكن مضمن فراغ repack_k4_quants(uint8_t * tile_dst, ثابت بلاک_T * b, صحيح لیکه, صحيح kt_local,
                                    int64_t sb_فی_لیکه, منطقي use_neon) {
#إذا defined(__ARM_NEON) || defined(__aarch64__)
    إذا (use_neon) {
        إذا ((لیکه & 7) == 0) {
            ثابت uint8_t * qs[8];
            لكل (صحيح rr = 0; rr < 8; ++rr) {
                qs[rr] = b[rr * sb_فی_لیکه].qs + (kt_local / 2) * 32;
            }
            repack_neon_k4_لیکې(qs, tile_dst, لیکه, (kt_local & 1) != 0);
        }
        أرجع;
    }
#وإلا
    GGML_ناکارول_شوی(sb_فی_لیکه);
    GGML_ناکارول_شوی(use_neon);
#د_که_پای
    ثابت uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
    ثابت صحيح ښويدنه = (kt_local & 1) ? 4 : 0;

    لكل (صحيح cp = 0; cp < 16; cp++) {
        ثابت uint8_t q0 = (qs_sub[2 * cp + 0] >> ښويدنه) & 0x0F;
        ثابت uint8_t q1 = (qs_sub[2 * cp + 1] >> ښويدنه) & 0x0F;
        tile_dst[cp * 32 + لیکه] = (uint8_t) ((q1 << 4) | q0);
    }
}

ساكن فراغ repack_q4_K_tiled(ggml_tensor * t, ثابت فراغ * معلومات, size_t دننه_واټن, size_t size) {
    ثابت int64_t sb_فی_لیکه = t->ne[0] / QK_K;
    ثابت منطقي    use_neon   = (t->ne[1] % 8 == 0);
    repack_kquant_منزل_tiled_grid<بلاک_q4_K>(t, معلومات, دننه_واټن, size, HTP_TENSOR_TILE_اندازه_Q4_K,
        [sb_فی_لیکه, use_neon](uint8_t * tile_dst, ثابت بلاک_q4_K * b, صحيح لیکه, صحيح kt_local) {
            ثابت عائم d    = GGML_FP16_منزل_FP32(b->d);
            ثابت عائم dmin = GGML_FP16_منزل_FP32(b->dmin);

            uint8_t sc, m;
            get_scale_min_k4(kt_local, b->scales, &sc, &m);

            repack_k4_quants(tile_dst, b, لیکه, kt_local, sb_فی_لیکه, use_neon);

            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
            scale_dst[2 * لیکه + 0] = GGML_FP32_منزل_FP16(d * (عائم) sc);
            scale_dst[2 * لیکه + 1] = GGML_FP32_منزل_FP16(-dmin * (عائم) m);
        });
}

ساكن فراغ repack_tiled_q4_K(فراغ * معلومات, ثابت ggml_tensor * t, size_t دننه_واټن, size_t size) {
    unpack_kquant_from_tiled_grid<بلاک_q4_K>(معلومات, t, دننه_واټن, size, HTP_TENSOR_TILE_اندازه_Q4_K,
        [](بلاک_q4_K * b, ثابت uint8_t * sb_tiles, size_t tile_اندازه, صحيح لیکه) {
            عائم sub_scales[8];
            عائم sub_mins[8];

            لكل (صحيح kt_local = 0; kt_local < 8; kt_local++) {
                ثابت uint8_t *   tile_src  = sb_tiles + kt_local * tile_اندازه;
                ثابت ggml_half * scale_src = (ثابت ggml_half *) (tile_src + 512);

                uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
                ثابت صحيح ښويدنه  = (kt_local & 1) ? 4 : 0;

                لكل (صحيح cp = 0; cp < 16; cp++) {
                    ثابت uint8_t val = tile_src[cp * 32 + لیکه];
                    qs_sub[2 * cp + 0] |= (uint8_t) ((val & 0x0F) << ښويدنه);
                    qs_sub[2 * cp + 1] |= (uint8_t) ((val >> 4)   << ښويدنه);
                }

                ثابت عائم D = GGML_FP16_منزل_FP32(scale_src[2 * لیکه + 0]);
                ثابت عائم M = GGML_FP16_منزل_FP32(scale_src[2 * لیکه + 1]);
                sub_scales[kt_local] = (D > 0.0f) ? D : 0.0f;
                sub_mins[kt_local]   = (-M > 0.0f) ? -M : 0.0f;
            }

            pack_k4_scales_mins(b->scales, &b->d, &b->dmin, sub_scales, sub_mins);
        });
}

ساكن فراغ repack_q5_K_tiled(ggml_tensor * t, ثابت فراغ * معلومات, size_t دننه_واټن, size_t size) {
    ثابت int64_t sb_فی_لیکه = t->ne[0] / QK_K;
    ثابت منطقي    use_neon   = (t->ne[1] % 8 == 0);
    repack_kquant_منزل_tiled_grid<بلاک_q5_K>(t, معلومات, دننه_واټن, size, HTP_TENSOR_TILE_اندازه_Q5_K,
        [sb_فی_لیکه, use_neon](uint8_t * tile_dst, ثابت بلاک_q5_K * b, صحيح لیکه, صحيح kt_local) {
            ثابت عائم d     = GGML_FP16_منزل_FP32(b->d);
            ثابت عائم dmin  = GGML_FP16_منزل_FP32(b->dmin);
            uint8_t * plane   = tile_dst + 640;

            uint8_t sc, m;
            get_scale_min_k4(kt_local, b->scales, &sc, &m);

            repack_k4_quants(tile_dst, b, لیکه, kt_local, sb_فی_لیکه, use_neon);

            ثابت uint8_t hbit = (uint8_t) (1 << kt_local);
#إذا defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q5_k_qh(b->qh, hbit, plane, لیکه);
#وإلا
            لكل (صحيح cp = 0; cp < 16; cp++) {
                ثابت صحيح i    = cp / 4;
                ثابت صحيح lane = (cp % 4) * 32 + لیکه;
                إذا (b->qh[2 * cp + 0] & hbit) {
                    plane[lane] |= (uint8_t) (1 << (2 * i));
                }
                إذا (b->qh[2 * cp + 1] & hbit) {
                    plane[lane] |= (uint8_t) (1 << (2 * i + 1));
                }
            }
#د_که_پای

            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
            scale_dst[2 * لیکه + 0] = GGML_FP32_منزل_FP16(d * (عائم) sc);
            scale_dst[2 * لیکه + 1] = GGML_FP32_منزل_FP16(-dmin * (عائم) m);
        });
}

ساكن فراغ repack_tiled_q5_K(فراغ * معلومات, ثابت ggml_tensor * t, size_t دننه_واټن, size_t size) {
    unpack_kquant_from_tiled_grid<بلاک_q5_K>(معلومات, t, دننه_واټن, size, HTP_TENSOR_TILE_اندازه_Q5_K,
        [](بلاک_q5_K * b, ثابت uint8_t * sb_tiles, size_t tile_اندازه, صحيح لیکه) {
            عائم sub_scales[8];
            عائم sub_mins[8];

            لكل (صحيح kt_local = 0; kt_local < 8; kt_local++) {
                ثابت uint8_t *   tile_src  = sb_tiles + kt_local * tile_اندازه;
                ثابت uint8_t *   plane     = tile_src + 640;
                ثابت ggml_half * scale_src = (ثابت ggml_half *) (tile_src + 512);

                uint8_t * qs_sub   = b->qs + (kt_local / 2) * 32;
                ثابت صحيح ښويدنه    = (kt_local & 1) ? 4 : 0;
                ثابت uint8_t hbit = (uint8_t) (1 << kt_local);

                لكل (صحيح cp = 0; cp < 16; cp++) {
                    ثابت uint8_t val = tile_src[cp * 32 + لیکه];
                    qs_sub[2 * cp + 0] |= (uint8_t) ((val & 0x0F) << ښويدنه);
                    qs_sub[2 * cp + 1] |= (uint8_t) ((val >> 4)   << ښويدنه);

                    ثابت صحيح i    = cp / 4;
                    ثابت صحيح lane = (cp % 4) * 32 + لیکه;
                    إذا (plane[lane] & (1 << (2 * i))) {
                        b->qh[2 * cp + 0] |= hbit;
                    }
                    إذا (plane[lane] & (1 << (2 * i + 1))) {
                        b->qh[2 * cp + 1] |= hbit;
                    }
                }

                ثابت عائم D = GGML_FP16_منزل_FP32(scale_src[2 * لیکه + 0]);
                ثابت عائم M = GGML_FP16_منزل_FP32(scale_src[2 * لیکه + 1]);
                sub_scales[kt_local] = (D > 0.0f) ? D : 0.0f;
                sub_mins[kt_local]   = (-M > 0.0f) ? -M : 0.0f;
            }

            pack_k4_scales_mins(b->scales, &b->d, &b->dmin, sub_scales, sub_mins);
        });
}

ساكن مضمن فراغ q1_0_tiled_بټ_location(صحيح k, صحيح لیکه, صحيح & byte_دننه_واټن, صحيح & بټ) {
    ثابت صحيح quarter = k / 32;
    ثابت صحيح kq      = k % 32;

    byte_دننه_واټن = quarter * 128 + 4 * لیکه + (kq % 4);
    بټ         = kq / 4;
}

ساكن فراغ repack_q1_0_tiled(ggml_tensor * t, ثابت فراغ * معلومات, size_t دننه_واټن, size_t size) {
    ثابت بلاک_q1_0 * src_ماتریکس = (ثابت بلاک_q1_0 *) معلومات;
    ثابت int64_t ne0 = t->ne[0];
    ثابت int64_t ne1 = t->ne[1];
    ثابت int64_t ne2 = t->ne[2];
    ثابت int64_t ne3 = t->ne[3];
    ثابت صحيح n_ستون_tiles = hex_پړاو_پورته(ne1, 32) / 32;
    ثابت صحيح n_k_tiles = ne0 / QK1_0;
    ثابت size_t tile_اندازه = HTP_TENSOR_TILE_اندازه_Q1_0;
    ثابت size_t ماتریکس_اندازه = (size_t) n_ستون_tiles * n_k_tiles * tile_اندازه;
    ثابت size_t ټوټه_اندازه = ne1 * ggml_لیکه_اندازه(t->type, ne0);
    ثابت int64_t پیل_ټوټه = دننه_واټن / ټوټه_اندازه;
    ثابت int64_t پای_ټکی_ټوټه = (std::min)((int64_t) ((دننه_واټن + size + ټوټه_اندازه - 1) / ټوټه_اندازه), ne2 * ne3);

    GGML_ASSERT(ne0 % QK1_0 == 0);

    لكل (int64_t ټوټه_idx = پیل_ټوټه; ټوټه_idx < پای_ټکی_ټوټه; ټوټه_idx++) {
        ثابت بلاک_q1_0 * src_ټوټه = src_ماتریکس + (ټوټه_idx - پیل_ټوټه) * ne1 * n_k_tiles;
        uint8_t * ماتریکس_dst = (uint8_t *) t->معلومات + ټوټه_idx * ماتریکس_اندازه;

        repack_parallel_for_ct(ماتریکس_اندازه, n_ستون_tiles, [&](صحيح ct0, صحيح ct1) {
            لكل (صحيح ct = ct0; ct < ct1; ct++) {
                لكل (صحيح kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = ماتریکس_dst + ((size_t) ct * n_k_tiles + kt) * tile_اندازه;
                    ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
                    memset(tile_dst, 0, tile_اندازه);

                    لكل (صحيح لیکه = 0; لیکه < 32; لیکه++) {
                        ثابت int64_t r = ct * 32 + لیکه;
                        إذا (r >= ne1) {
                            استمر;
                        }

                        ثابت بلاک_q1_0 & بلاک = src_ټوټه[r * n_k_tiles + kt];
                        scale_dst[لیکه] = بلاک.d;
#إذا defined(__ARM_NEON) || defined(__aarch64__)
                        repack_neon_q1_0_بلاک(بلاک.qs, tile_dst, لیکه);
#وإلا
                        لكل (صحيح k = 0; k < QK1_0; k++) {
                            صحيح byte_دننه_واټن;
                            صحيح بټ;
                            q1_0_tiled_بټ_location(k, لیکه, byte_دننه_واټن, بټ);
                            إذا (بلاک.qs[k / 8] & (1u << (k % 8))) {
                                tile_dst[byte_دننه_واټن] |= (uint8_t) (1u << بټ);
                            }
                        }
#د_که_پای
                    }
                }
            }
        });
    }
}

ساكن فراغ repack_tiled_q1_0(فراغ * معلومات, ثابت ggml_tensor * t, size_t دننه_واټن, size_t size) {
    بلاک_q1_0 * dst_ماتریکس = (بلاک_q1_0 *) معلومات;
    ثابت int64_t ne0 = t->ne[0];
    ثابت int64_t ne1 = t->ne[1];
    ثابت int64_t ne2 = t->ne[2];
    ثابت int64_t ne3 = t->ne[3];
    ثابت صحيح n_ستون_tiles = hex_پړاو_پورته(ne1, 32) / 32;
    ثابت صحيح n_k_tiles = ne0 / QK1_0;
    ثابت size_t tile_اندازه = HTP_TENSOR_TILE_اندازه_Q1_0;
    ثابت size_t ماتریکس_اندازه = (size_t) n_ستون_tiles * n_k_tiles * tile_اندازه;
    ثابت size_t لیکه_اندازه_بایټونه = ggml_لیکه_اندازه(t->type, ne0);
    ثابت size_t ټوټه_اندازه = ne1 * لیکه_اندازه_بایټونه;
    ثابت int64_t پیل_ټوټه = دننه_واټن / ټوټه_اندازه;
    ثابت int64_t پای_ټکی_ټوټه = (std::min)((int64_t) ((دننه_واټن + size + ټوټه_اندازه - 1) / ټوټه_اندازه), ne2 * ne3);

    GGML_ASSERT(ne0 % QK1_0 == 0);

    لكل (int64_t ټوټه_idx = پیل_ټوټه; ټوټه_idx < پای_ټکی_ټوټه; ټوټه_idx++) {
        ثابت size_t cur_پیل_byte = (std::max)(دننه_واټن, (size_t) ټوټه_idx * ټوټه_اندازه);
        ثابت size_t cur_پای_ټکی_byte = (std::min)(دننه_واټن + size, (size_t) (ټوټه_idx + 1) * ټوټه_اندازه);
        ثابت int64_t پیل_لیکه = (cur_پیل_byte - (size_t) ټوټه_idx * ټوټه_اندازه) / لیکه_اندازه_بایټونه;
        ثابت int64_t پای_ټکی_لیکه = (std::min)((int64_t) ((cur_پای_ټکی_byte - (size_t) ټوټه_idx * ټوټه_اندازه + لیکه_اندازه_بایټونه - 1) / لیکه_اندازه_بایټونه), ne1);
        بلاک_q1_0 * dst_ټوټه = dst_ماتریکس + (cur_پیل_byte - دننه_واټن) / حجم(بلاک_q1_0);
        ثابت uint8_t * ماتریکس_src = (ثابت uint8_t *) t->معلومات + ټوټه_idx * ماتریکس_اندازه;

        لكل (صحيح ct = پیل_لیکه / 32; ct < (پای_ټکی_لیکه + 31) / 32; ct++) {
            لكل (صحيح kt = 0; kt < n_k_tiles; kt++) {
                ثابت uint8_t * tile_src = ماتریکس_src + ((size_t) ct * n_k_tiles + kt) * tile_اندازه;
                ثابت ggml_half * scale_src = (ثابت ggml_half *) (tile_src + 512);

                لكل (صحيح لیکه = 0; لیکه < 32; لیکه++) {
                    ثابت int64_t r = ct * 32 + لیکه;
                    إذا (r < پیل_لیکه || r >= پای_ټکی_لیکه) {
                        استمر;
                    }

                    بلاک_q1_0 & بلاک = dst_ټوټه[(r - پیل_لیکه) * n_k_tiles + kt];
                    memset(بلاک.qs, 0, حجم(بلاک.qs));
                    بلاک.d = scale_src[لیکه];
                    لكل (صحيح k = 0; k < QK1_0; k++) {
                        صحيح byte_دننه_واټن;
                        صحيح بټ;
                        q1_0_tiled_بټ_location(k, لیکه, byte_دننه_واټن, بټ);
                        إذا (tile_src[byte_دننه_واټن] & (1u << بټ)) {
                            بلاک.qs[k / 8] |= (uint8_t) (1u << (k % 8));
                        }
                    }
                }
            }
        }
    }
}

// --- Top-level Repack / Unpack Dispatchers ---

فراغ htp_repack_tensor_tiled(ggml_tensor * tensor, ثابت فراغ * معلومات, size_t size) {
    بدل (tensor->type) {
        حالة GGML_TYPE_Q1_0:
            repack_q1_0_tiled(tensor, معلومات, 0, size);
            اكسر;
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_IQ4_NL:
            repack_q4_0_tiled(tensor, معلومات, 0, size);
            اكسر;
        حالة GGML_TYPE_Q4_1:
            repack_q4_1_tiled(tensor, معلومات, 0, size);
            اكسر;
        حالة GGML_TYPE_Q8_0:
            repack_q8_0_tiled(tensor, معلومات, 0, size);
            اكسر;
        حالة GGML_TYPE_MXFP4:
            repack_mxfp4_tiled(tensor, معلومات, 0, size);
            اكسر;
        حالة GGML_TYPE_Q2_0:
            repack_q2_0_tiled(tensor, معلومات, 0, size);
            اكسر;
        حالة GGML_TYPE_Q2_K:
            repack_q2_K_tiled(tensor, معلومات, 0, size);
            اكسر;
        حالة GGML_TYPE_Q3_K:
            repack_q3_K_tiled(tensor, معلومات, 0, size);
            اكسر;
        حالة GGML_TYPE_Q4_K:
            repack_q4_K_tiled(tensor, معلومات, 0, size);
            اكسر;
        حالة GGML_TYPE_Q5_K:
            repack_q5_K_tiled(tensor, معلومات, 0, size);
            اكسر;
        حالة GGML_TYPE_Q6_K:
            repack_q6_K_tiled(tensor, معلومات, 0, size);
            اكسر;
        حالة GGML_TYPE_F16:
            repack_float_منزل_f16_tiled<uint16_t>(tensor, معلومات, 0, size,
                [](uint16_t v) { أرجع v; });
            اكسر;
        حالة GGML_TYPE_F32:
            repack_float_منزل_f16_tiled<عائم>(tensor, معلومات, 0, size,
                [](عائم v) { أرجع (uint16_t) GGML_FP32_منزل_FP16(v); });
            اكسر;
        حالة GGML_TYPE_BF16:
            repack_float_منزل_f16_tiled<ggml_bf16_t>(tensor, معلومات, 0, size,
                [](ggml_bf16_t v) { أرجع (uint16_t) GGML_FP32_منزل_FP16(GGML_BF16_منزل_FP32(v)); });
            اكسر;
        افتراضي:
            اكسر;
    }
}

فراغ htp_unpack_tensor_tiled(فراغ * معلومات, ثابت ggml_tensor * tensor, size_t دننه_واټن, size_t size) {
    GGML_ASSERT(دننه_واټن + size <= ggml_nbytes(tensor));

    بدل (tensor->type) {
        حالة GGML_TYPE_Q1_0:
            repack_tiled_q1_0(معلومات, tensor, دننه_واټن, size);
            اكسر;
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_IQ4_NL:
            repack_tiled_q4_0(معلومات, tensor, دننه_واټن, size);
            اكسر;
        حالة GGML_TYPE_Q4_1:
            repack_tiled_q4_1(معلومات, tensor, دننه_واټن, size);
            اكسر;
        حالة GGML_TYPE_Q8_0:
            repack_tiled_q8_0(معلومات, tensor, دننه_واټن, size);
            اكسر;
        حالة GGML_TYPE_MXFP4:
            repack_tiled_mxfp4(معلومات, tensor, دننه_واټن, size);
            اكسر;
        حالة GGML_TYPE_Q2_0:
            repack_tiled_q2_0(معلومات, tensor, دننه_واټن, size);
            اكسر;
        حالة GGML_TYPE_Q2_K:
            repack_tiled_q2_K(معلومات, tensor, دننه_واټن, size);
            اكسر;
        حالة GGML_TYPE_Q3_K:
            repack_tiled_q3_K(معلومات, tensor, دننه_واټن, size);
            اكسر;
        حالة GGML_TYPE_Q4_K:
            repack_tiled_q4_K(معلومات, tensor, دننه_واټن, size);
            اكسر;
        حالة GGML_TYPE_Q5_K:
            repack_tiled_q5_K(معلومات, tensor, دننه_واټن, size);
            اكسر;
        حالة GGML_TYPE_Q6_K:
            repack_tiled_q6_K(معلومات, tensor, دننه_واټن, size);
            اكسر;
        حالة GGML_TYPE_F16:
            unpack_f16_tiled_منزل_float<uint16_t>(معلومات, tensor, دننه_واټن, size,
                [](uint16_t v) { أرجع v; });
            اكسر;
        حالة GGML_TYPE_F32:
            unpack_f16_tiled_منزل_float<عائم>(معلومات, tensor, دننه_واټن, size,
                [](uint16_t v) { أرجع ggml_fp16_منزل_fp32((ggml_fp16_t) v); });
            اكسر;
        حالة GGML_TYPE_BF16:
            unpack_f16_tiled_منزل_float<ggml_bf16_t>(معلومات, tensor, دننه_واټن, size,
                [](uint16_t v) { أرجع ggml_fp32_منزل_bf16(ggml_fp16_منزل_fp32((ggml_fp16_t) v)); });
            اكسر;
        افتراضي:
            حافظه_نقل(معلومات, (ثابت حرف *) tensor->معلومات + دننه_واټن, size);
            اكسر;
    }
}
