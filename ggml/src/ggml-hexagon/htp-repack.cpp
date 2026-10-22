#ضمن "htp-repack.h"

#وصف_ڪريو GGML_COMMON_IMPL_CPP
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
#جيڪڏهن_پورو

// --- 4-bit block-32 nibble helpers (shared by Q4_0, IQ4_NL, Q4_1, MXFP4) ---

ساكن مضمن فراغ unpack_nibbles_32(uint8_t * dst, ثابت uint8_t * qs) {
    لكل (بنانشان صحيح i = 0; i < 16; ++i) {
        dst[i + 0]  = qs[i] & 0x0F;
        dst[i + 16] = qs[i] >> 4;
    }
}

ساكن مضمن فراغ pack_nibbles_32(uint8_t * qs, ثابت uint8_t * src) {
    لكل (بنانشان صحيح i = 0; i < 16; ++i) {
        qs[i] = src[i + 0] | (src[i + 16] << 4);
    }
}

ساكن مضمن فراغ transpose_nibbles_منزل_tile(uint8_t * tile_dst, ثابت uint8_t tile_quants[32][32]) {
    لكل (صحيح cp = 0; cp < 16; cp++) {
        لكل (صحيح صف = 0; صف < 32; صف++) {
            tile_dst[cp * 32 + صف] = (tile_quants[صف][2 * cp + 1] << 4) | tile_quants[صف][2 * cp];
        }
    }
}

ساكن مضمن فراغ transpose_tile_منزل_nibbles(uint8_t tile_quants[32][32], ثابت uint8_t * tile_src) {
    لكل (صحيح cp = 0; cp < 16; cp++) {
        لكل (صحيح صف = 0; صف < 32; صف++) {
            ثابت uint8_t val = tile_src[cp * 32 + صف];
            tile_quants[صف][2 * cp + 0] = val & 0x0F;
            tile_quants[صف][2 * cp + 1] = val >> 4;
        }
    }
}

#إذا defined(__ARM_NEON) || defined(__aarch64__)
ساكن فراغ repack_neon_transpose8x8(ثابت uint8x8_t صفون[8], uint8x8_t columns[8]) {
    ثابت uint8x8x2_t t0 = vtrn_u8(صفون[0], صفون[1]);
    ثابت uint8x8x2_t t1 = vtrn_u8(صفون[2], صفون[3]);
    ثابت uint8x8x2_t t2 = vtrn_u8(صفون[4], صفون[5]);
    ثابت uint8x8x2_t t3 = vtrn_u8(صفون[6], صفون[7]);
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

ساكن uint8x8_t repack_neon_pack_4bit_جوڙا(uint8x8_t پهريون, uint8x8_t second, منطقي مٿاهون) {
    ثابت uint8x8x2_t separated = vuzp_u8(پهريون, second);
    ثابت uint8x8_t نقاب = vdup_n_u8(0x0F);
    uint8x8_t even = separated.val[0];
    uint8x8_t odd  = separated.val[1];

    إذا (مٿاهون) {
        even = vshr_n_u8(even, 4);
        odd  = vshr_n_u8(odd, 4);
    }

    even = vand_u8(even, نقاب);
    odd  = vand_u8(odd, نقاب);
    أرجع vorr_u8(vshl_n_u8(odd, 4), even);
}

ساكن فراغ repack_neon_4bit_صفون(ثابت uint8_t * qs[8], uint8_t * tile_dst, صحيح row0) {
    uint8x8_t هيٺيون_صفون[8];
    uint8x8_t مٿاهون_صفون[8];
    uint8x8_t columns[8];

    لكل (صحيح rr = 0; rr < 8; ++rr) {
        ثابت uint8x16_t packed = vld1q_u8(qs[rr]);
        ثابت uint8x8_t پهريون = vget_هيٺيون_u8(packed);
        ثابت uint8x8_t second = vget_مٿاهون_u8(packed);
        هيٺيون_صفون[rr]  = repack_neon_pack_4bit_جوڙا(پهريون, second, خطأ);
        مٿاهون_صفون[rr] = repack_neon_pack_4bit_جوڙا(پهريون, second, سچ);
    }

    repack_neon_transpose8x8(هيٺيون_صفون, columns);
    لكل (صحيح cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + cp * 32 + row0, columns[cp]);
    }

    repack_neon_transpose8x8(مٿاهون_صفون, columns);
    لكل (صحيح cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + (cp + 8) * 32 + row0, columns[cp]);
    }
}

ساكن فراغ repack_neon_k4_صفون(ثابت uint8_t * qs[8], uint8_t * tile_dst, صحيح row0, منطقي مٿاهون) {
    uint8x8_t پهريون_صفون[8];
    uint8x8_t ٻيو_صفون[8];
    uint8x8_t columns[8];

    لكل (صحيح rr = 0; rr < 8; ++rr) {
        پهريون_صفون[rr] = repack_neon_pack_4bit_جوڙا(vld1_u8(qs[rr]), vld1_u8(qs[rr] + 8), مٿاهون);
        ٻيو_صفون[rr] = repack_neon_pack_4bit_جوڙا(vld1_u8(qs[rr] + 16), vld1_u8(qs[rr] + 24), مٿاهون);
    }

    repack_neon_transpose8x8(پهريون_صفون, columns);
    لكل (صحيح cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + cp * 32 + row0, columns[cp]);
    }

    repack_neon_transpose8x8(ٻيو_صفون, columns);
    لكل (صحيح cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + (cp + 8) * 32 + row0, columns[cp]);
    }
}

ساكن فراغ repack_neon_q8_صفون(ثابت uint8_t * qs[8], uint8_t * tile_dst, صحيح row0) {
    uint8x8_t صفون[8];
    uint8x8_t columns[8];

    لكل (صحيح cp = 0; cp < 16; cp += 4) {
        لكل (صحيح rr = 0; rr < 8; ++rr) {
            صفون[rr] = vld1_u8(qs[rr] + 2 * cp);
        }
        repack_neon_transpose8x8(صفون, columns);

        لكل (صحيح pair = 0; pair < 4; ++pair) {
            ثابت uint8x8x2_t interleaved = vzip_u8(columns[2 * pair], columns[2 * pair + 1]);
            uint8_t * dst = tile_dst + (cp + pair) * 64 + 2 * row0;
            vst1_u8(dst, interleaved.val[0]);
            vst1_u8(dst + 8, interleaved.val[1]);
        }
    }
}

ساكن مضمن فراغ repack_store_u32(uint8_t * dst, uint32_t v) {
    يادگيري_نقل(dst, &v, حجم(v));
}

ساكن مضمن uint32x4_t repack_neon_transpose_2bit_4x4(uint32x4_t w) {
    uint32x4_t t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 6)), vdupq_n_u32(0x00CC00CCu));
    w = veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 6)));
    t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 12)), vdupq_n_u32(0x0000F0F0u));
    أرجع veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 12)));
}

ساكن مضمن فراغ repack_neon_q2_0_بلاڪ(ثابت uint8_t * qs, uint8_t * tile, صحيح صف) {
    ثابت uint32x4_t w = repack_neon_transpose_2bit_4x4(vreinterpretq_u32_u8(vld1q_u8(qs)));
    repack_store_u32(tile + 0 * 128 + 4 * صف, vgetq_lane_u32(w, 0));
    repack_store_u32(tile + 1 * 128 + 4 * صف, vgetq_lane_u32(w, 1));
    repack_store_u32(tile + 2 * 128 + 4 * صف, vgetq_lane_u32(w, 2));
    repack_store_u32(tile + 3 * 128 + 4 * صف, vgetq_lane_u32(w, 3));
}

ساكن مضمن فراغ repack_neon_q1_0_بلاڪ(ثابت uint8_t * qs, uint8_t * tile, صحيح صف) {
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

ساكن مضمن فراغ repack_neon_pack_2bit_plane(uint8x16_t v0, uint8x16_t v1, uint8_t * dst, صحيح صف) {
    ثابت uint32x4x2_t z   = vuzpq_u32(vreinterpretq_u32_u8(v0), vreinterpretq_u32_u8(v1));
    ثابت uint32x4_t   p01 = vorrq_u32(z.val[0], vshlq_n_u32(z.val[1], 2));
    ثابت uint32x2x2_t u   = vuzp_u32(vget_هيٺيون_u32(p01), vget_مٿاهون_u32(p01));
    ثابت uint32x2_t   ٻاهريون = vorr_u32(u.val[0], vshl_n_u32(u.val[1], 4));
    repack_store_u32(dst + 0 * 128 + صف * 4, vget_lane_u32(ٻاهريون, 0));
    repack_store_u32(dst + 1 * 128 + صف * 4, vget_lane_u32(ٻاهريون, 1));
}

ساكن مضمن فراغ repack_neon_q2_3_k_low2(ثابت uint8_t * qs, صحيح kt_local, uint8_t * lo_pl, صحيح صف) {
    ثابت uint8_t *  qs_sub = qs + (kt_local / 4) * 32;
    ثابت int8x16_t  سرڪايو  = vdupq_n_s8(-2 * (kt_local % 4));
    ثابت uint8x16_t نقاب   = vdupq_n_u8(0x03);
    ثابت uint8x16_t v0     = vandq_u8(vshlq_u8(vld1q_u8(qs_sub +  0), سرڪايو), نقاب);
    ثابت uint8x16_t v1     = vandq_u8(vshlq_u8(vld1q_u8(qs_sub + 16), سرڪايو), نقاب);
    repack_neon_pack_2bit_plane(v0, v1, lo_pl, صف);
}

ساكن مضمن فراغ repack_neon_q3_k_hmask(ثابت uint8_t * hmask, صحيح kt_local, uint8_t * neg_pl, صحيح صف) {
    ساكن ثابت uint32_t نقاب_لفظ[4] = { 0x01010101u, 0x02020202u, 0x04040404u, 0x08080808u };
    ثابت uint8x16_t m0   = vreinterpretq_u8_u32(vld1q_u32(نقاب_لفظ));
    ثابت uint8x16_t m1   = vshlq_n_u8(m0, 4);
    ثابت uint8x16_t hbit = vdupq_n_u8((uint8_t) (1u << kt_local));
    ثابت uint8x16_t b0   = vbicq_u8(m0, vtstq_u8(vld1q_u8(hmask +  0), hbit));
    ثابت uint8x16_t b1   = vbicq_u8(m1, vtstq_u8(vld1q_u8(hmask + 16), hbit));
    ثابت uint32x4_t b01  = vreinterpretq_u32_u8(vorrq_u8(b0, b1));
    ثابت uint32x2_t f0   = vorr_u32(vget_هيٺيون_u32(b01), vget_مٿاهون_u32(b01));
    ثابت uint32x2_t f1   = vpadd_u32(f0, f0);
    repack_store_u32(neg_pl + صف * 4, vget_lane_u32(f1, 0));
}

ساكن مضمن فراغ repack_neon_q6_k(ثابت بلاڪ_q6_K * b, صحيح kt_local, uint8_t * lo_pl, uint8_t * hi_pl, صحيح صف) {
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
    repack_store_u32(lo_pl + 0 * 128 + صف * 4, vgetq_lane_u32(lout, 0));
    repack_store_u32(lo_pl + 1 * 128 + صف * 4, vgetq_lane_u32(lout, 1));
    repack_store_u32(lo_pl + 2 * 128 + صف * 4, vgetq_lane_u32(lout, 2));
    repack_store_u32(lo_pl + 3 * 128 + صف * 4, vgetq_lane_u32(lout, 3));

    ثابت uint8x16_t hi0 = vandq_u8(vshlq_u8(vld1q_u8(qh +  0), hshift), mask2);
    ثابت uint8x16_t hi1 = vandq_u8(vshlq_u8(vld1q_u8(qh + 16), hshift), mask2);
    repack_neon_pack_2bit_plane(hi0, hi1, hi_pl, صف);
}

ساكن مضمن فراغ repack_neon_q5_k_qh(ثابت uint8_t * qh, uint8_t hbit, uint8_t * plane, صحيح صف) {
    ساكن ثابت uint8_t نقاب_بائيٽون[16] = { 1, 2, 1, 2, 1, 2, 1, 2, 4, 8, 4, 8, 4, 8, 4, 8 };
    ثابت uint8x16_t m0   = vld1q_u8(نقاب_بائيٽون);
    ثابت uint8x16_t m1   = vshlq_n_u8(m0, 4);
    ثابت uint8x16_t vbit = vdupq_n_u8(hbit);
    ثابت uint8x16_t b0   = vandq_u8(vtstq_u8(vld1q_u8(qh +  0), vbit), m0);
    ثابت uint8x16_t b1   = vandq_u8(vtstq_u8(vld1q_u8(qh + 16), vbit), m1);
    ثابت uint8x16_t b01  = vorrq_u8(b0, b1);
    ثابت uint8x8_t  fold = vorr_u8(vget_هيٺيون_u8(b01), vget_مٿاهون_u8(b01));
    ثابت uint8x8_t  ٻاهريون  = vpadd_u8(fold, fold);
    plane[0 * 32 + صف] = vget_lane_u8(ٻاهريون, 0);
    plane[1 * 32 + صف] = vget_lane_u8(ٻاهريون, 1);
    plane[2 * 32 + صف] = vget_lane_u8(ٻاهريون, 2);
    plane[3 * 32 + صف] = vget_lane_u8(ٻاهريون, 3);
}

ساكن مضمن uint16x8_t repack_neon_ڀريو_f16x8(ثابت uint16_t * src) {
    أرجع vld1q_u16(src);
}

#إذا defined(__aarch64__) || defined(__ARM_FP16_FORMAT_IEEE)
ساكن مضمن uint16x8_t repack_neon_ڀريو_f16x8(ثابت عائم * src) {
    ثابت float16x4_t lo = vcvt_f16_f32(vld1q_f32(src));
    ثابت float16x4_t hi = vcvt_f16_f32(vld1q_f32(src + 4));
    أرجع vreinterpretq_u16_f16(vcombine_f16(lo, hi));
}

ساكن مضمن uint16x8_t repack_neon_ڀريو_f16x8(ثابت ggml_bf16_t * src) {
    ثابت uint16x8_t  bf   = vld1q_u16((ثابت uint16_t *) src);
    ثابت float32x4_t f_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_هيٺيون_u16(bf), 16));
    ثابت float32x4_t f_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_مٿاهون_u16(bf), 16));
    ثابت float16x4_t lo   = vcvt_f16_f32(f_lo);
    ثابت float16x4_t hi   = vcvt_f16_f32(f_hi);
    أرجع vreinterpretq_u16_f16(vcombine_f16(lo, hi));
}
#وإلا
ساكن مضمن uint16x8_t repack_neon_ڀريو_f16x8(ثابت عائم * src) {
    uint16_t tmp[8];
    لكل (صحيح i = 0; i < 8; ++i) {
        tmp[i] = GGML_FP32_منزل_FP16(src[i]);
    }
    أرجع vld1q_u16(tmp);
}

ساكن مضمن uint16x8_t repack_neon_ڀريو_f16x8(ثابت ggml_bf16_t * src) {
    uint16_t tmp[8];
    لكل (صحيح i = 0; i < 8; ++i) {
        tmp[i] = GGML_FP32_منزل_FP16(GGML_BF16_منزل_FP32(src[i]));
    }
    أرجع vld1q_u16(tmp);
}
#جيڪڏهن_پورو

سانچو <قسم_نالو SrcT>
ساكن فراغ repack_neon_f16_tile(uint16_t * tile_dst, ثابت SrcT * src_ٽڪرو, int64_t صف_stride, صحيح ct, صحيح kt) {
    لكل (صحيح row0 = 0; row0 < 32; row0 += 4) {
        ثابت SrcT * r0 = src_ٽڪرو + (int64_t) (ct * 32 + row0 + 0) * صف_stride + kt * 32;
        ثابت SrcT * r1 = src_ٽڪرو + (int64_t) (ct * 32 + row0 + 1) * صف_stride + kt * 32;
        ثابت SrcT * r2 = src_ٽڪرو + (int64_t) (ct * 32 + row0 + 2) * صف_stride + kt * 32;
        ثابت SrcT * r3 = src_ٽڪرو + (int64_t) (ct * 32 + row0 + 3) * صف_stride + kt * 32;
        لكل (صحيح cp = 0; cp < 16; cp += 4) {
            ثابت uint32x4_t v0 = vreinterpretq_u32_u16(repack_neon_ڀريو_f16x8(r0 + 2 * cp));
            ثابت uint32x4_t v1 = vreinterpretq_u32_u16(repack_neon_ڀريو_f16x8(r1 + 2 * cp));
            ثابت uint32x4_t v2 = vreinterpretq_u32_u16(repack_neon_ڀريو_f16x8(r2 + 2 * cp));
            ثابت uint32x4_t v3 = vreinterpretq_u32_u16(repack_neon_ڀريو_f16x8(r3 + 2 * cp));

            ثابت uint32x2x2_t t0 = vtrn_u32(vget_هيٺيون_u32(v0),  vget_هيٺيون_u32(v1));
            ثابت uint32x2x2_t t1 = vtrn_u32(vget_مٿاهون_u32(v0), vget_مٿاهون_u32(v1));
            ثابت uint32x2x2_t t2 = vtrn_u32(vget_هيٺيون_u32(v2),  vget_هيٺيون_u32(v3));
            ثابت uint32x2x2_t t3 = vtrn_u32(vget_مٿاهون_u32(v2), vget_مٿاهون_u32(v3));

            vst1q_u16(tile_dst + (cp + 0) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t0.val[0], t2.val[0])));
            vst1q_u16(tile_dst + (cp + 1) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t0.val[1], t2.val[1])));
            vst1q_u16(tile_dst + (cp + 2) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t1.val[0], t3.val[0])));
            vst1q_u16(tile_dst + (cp + 3) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t1.val[1], t3.val[1])));
        }
    }
}
#جيڪڏهن_پورو

// --- Generic 32x32 tile grid walkers for block-32 and float types ---

سانچو <قسم_نالو Fn>
ساكن فراغ repack_parallel_for_ct(size_t ٽڪرو_بائيٽون, صحيح n_کانو_tiles, Fn && fn) {
    صحيح n_ڌاڳا = 1;
    إذا (ٽڪرو_بائيٽون > 128 * 1024) {
        لكل (صحيح nt : { 4, 3, 2 }) {
            إذا (n_کانو_tiles % nt == 0) {
                n_ڌاڳا = nt;
                اكسر;
            }
        }
    }

    إذا (n_ڌاڳا <= 1) {
        fn(0, n_کانو_tiles);
        أرجع;
    }

    ثابت صحيح chunk = n_کانو_tiles / n_ڌاڳا;
    std::thread workers[3];
    لكل (صحيح i = 0; i < n_ڌاڳا - 1; ++i) {
        workers[i] = std::thread([&fn, i, chunk]() {
            fn(i * chunk, (i + 1) * chunk);
        });
    }
    fn((n_ڌاڳا - 1) * chunk, n_کانو_tiles);
    لكل (صحيح i = 0; i < n_ڌاڳا - 1; ++i) {
        workers[i].join();
    }
}

سانچو <قسم_نالو SrcElemT, صحيح ELEMS_في_KTILE, قسم_نالو PackTileFn>
ساكن فراغ repack_منزل_tiled_grid_32(ggml_tensor * t, ثابت فراغ * معلومات, size_t وٿي, size_t size,
                                    size_t tile_سائيز, PackTileFn && pack_tile) {
    ثابت SrcElemT * src_ميٽرڪس = (ثابت SrcElemT *) معلومات;
    ثابت int64_t ne0 = t->ne[0];
    ثابت int64_t ne1 = t->ne[1];
    ثابت int64_t ne2 = t->ne[2];
    ثابت int64_t ne3 = t->ne[3];
    ثابت صحيح n_کانو_tiles = hex_ڦيرو_مٿي(ne1, 32) / 32;
    ثابت صحيح n_k_tiles   = hex_ڦيرو_مٿي(ne0, 32) / 32;
    ثابت size_t ميٽرڪس_سائيز = (size_t) n_کانو_tiles * n_k_tiles * tile_سائيز;

    ثابت size_t ٽڪرو_سائيز = (size_t) ne1 * ggml_صف_سائيز(t->type, ne0);
    ثابت int64_t شروعات_ٽڪرو = وٿي / ٽڪرو_سائيز;
    int64_t پڇاڙي_ٽڪرو = (وٿي + size + ٽڪرو_سائيز - 1) / ٽڪرو_سائيز;
    إذا (پڇاڙي_ٽڪرو > ne2 * ne3) {
        پڇاڙي_ٽڪرو = ne2 * ne3;
    }

    ثابت int64_t صف_stride = (ne0 / (32 / ELEMS_في_KTILE));

    لكل (int64_t ٽڪرو_idx = شروعات_ٽڪرو; ٽڪرو_idx < پڇاڙي_ٽڪرو; ٽڪرو_idx++) {
        ثابت SrcElemT * src_ٽڪرو = src_ميٽرڪس + (ٽڪرو_idx - شروعات_ٽڪرو) * (ne1 * صف_stride);
        uint8_t * ميٽرڪس_dst = (uint8_t *) t->معلومات + ٽڪرو_idx * ميٽرڪس_سائيز;

        repack_parallel_for_ct(ميٽرڪس_سائيز, n_کانو_tiles, [&](صحيح ct0, صحيح ct1) {
            لكل (صحيح ct = ct0; ct < ct1; ct++) {
                لكل (صحيح kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = ميٽرڪس_dst + ((size_t) ct * n_k_tiles + kt) * tile_سائيز;
                    pack_tile(tile_dst, src_ٽڪرو, ne0, ne1, صف_stride, ct, kt);
                }
            }
        });
    }
}

سانچو <قسم_نالو DstElemT, صحيح ELEMS_في_KTILE, قسم_نالو UnpackTileFn>
ساكن فراغ unpack_from_tiled_grid_32(فراغ * معلومات, ثابت ggml_tensor * t, size_t وٿي, size_t size,
                                      size_t tile_سائيز, UnpackTileFn && unpack_tile) {
    uint8_t * dst_بائيٽون = (uint8_t *) معلومات;
    ثابت int64_t ne0 = t->ne[0];
    ثابت int64_t ne1 = t->ne[1];
    ثابت int64_t ne2 = t->ne[2];
    ثابت int64_t ne3 = t->ne[3];
    ثابت صحيح n_کانو_tiles = hex_ڦيرو_مٿي(ne1, 32) / 32;
    ثابت صحيح n_k_tiles   = hex_ڦيرو_مٿي(ne0, 32) / 32;
    ثابت size_t ميٽرڪس_سائيز = (size_t) n_کانو_tiles * n_k_tiles * tile_سائيز;

    ثابت size_t صف_سائيز_بائيٽون = ggml_صف_سائيز(t->type, ne0);
    ثابت size_t ٽڪرو_سائيز     = (size_t) ne1 * صف_سائيز_بائيٽون;
    ثابت int64_t شروعات_ٽڪرو   = وٿي / ٽڪرو_سائيز;
    int64_t پڇاڙي_ٽڪرو           = (وٿي + size + ٽڪرو_سائيز - 1) / ٽڪرو_سائيز;
    إذا (پڇاڙي_ٽڪرو > ne2 * ne3) {
        پڇاڙي_ٽڪرو = ne2 * ne3;
    }

    ثابت int64_t صف_stride = (ne0 / (32 / ELEMS_في_KTILE));

    لكل (int64_t ٽڪرو_idx = شروعات_ٽڪرو; ٽڪرو_idx < پڇاڙي_ٽڪرو; ٽڪرو_idx++) {
        ثابت size_t cur_شروعات_byte     = (std::max)(وٿي, (size_t) ٽڪرو_idx * ٽڪرو_سائيز);
        ثابت size_t cur_پڇاڙي_byte       = (std::min)(وٿي + size, (size_t) (ٽڪرو_idx + 1) * ٽڪرو_سائيز);
        ثابت size_t ٽڪرو_وٿي_شروعات = cur_شروعات_byte - (size_t) ٽڪرو_idx * ٽڪرو_سائيز;
        ثابت size_t ٽڪرو_وٿي_پڇاڙي   = cur_پڇاڙي_byte - (size_t) ٽڪرو_idx * ٽڪرو_سائيز;

        ثابت int64_t شروعات_صف = ٽڪرو_وٿي_شروعات / صف_سائيز_بائيٽون;
        ثابت int64_t پڇاڙي_صف   = (std::min)((int64_t) ((ٽڪرو_وٿي_پڇاڙي + صف_سائيز_بائيٽون - 1) / صف_سائيز_بائيٽون), ne1);

        ثابت صحيح شروعات_ct = (صحيح) (شروعات_صف / 32);
        ثابت صحيح پڇاڙي_ct   = (std::min)((صحيح) ((پڇاڙي_صف + 31) / 32), n_کانو_tiles);

        DstElemT * dst_ٽڪرو = (DstElemT *) (dst_بائيٽون + (cur_شروعات_byte - وٿي));
        ثابت uint8_t * ميٽرڪس_src = (ثابت uint8_t *) t->معلومات + ٽڪرو_idx * ميٽرڪس_سائيز;

        لكل (صحيح ct = شروعات_ct; ct < پڇاڙي_ct; ct++) {
            لكل (صحيح kt = 0; kt < n_k_tiles; kt++) {
                ثابت uint8_t * tile_src = ميٽرڪس_src + ((size_t) ct * n_k_tiles + kt) * tile_سائيز;
                unpack_tile(dst_ٽڪرو, tile_src, ne0, شروعات_صف, پڇاڙي_صف, صف_stride, ct, kt);
            }
        }
    }
}

// --- Block-32 Quantized Types (Q4_0, IQ4_NL, Q4_1, Q8_0, MXFP4) ---

ساكن فراغ repack_q4_0_tiled(ggml_tensor * t, ثابت فراغ * معلومات, size_t وٿي, size_t size) {
    repack_منزل_tiled_grid_32<بلاڪ_q4_0, 1>(t, معلومات, وٿي, size, HTP_TENSOR_TILE_سائيز_Q4_0,
        [](uint8_t * tile_dst, ثابت بلاڪ_q4_0 * src_ٽڪرو, int64_t ne0, int64_t ne1, int64_t صف_stride, صحيح ct, صحيح kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
#إذا defined(__ARM_NEON) || defined(__aarch64__)
            إذا (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                لكل (صحيح row0 = 0; row0 < 32; row0 += 8) {
                    ثابت uint8_t * qs[8];
                    لكل (صحيح rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_ٽڪرو[(ct * 32 + row0 + rr) * صف_stride + kt].qs;
                    }
                    repack_neon_4bit_صفون(qs, tile_dst, row0);
                }

                لكل (صحيح صف = 0; صف < 32; ++صف) {
                    scale_dst[صف] = src_ٽڪرو[(ct * 32 + صف) * صف_stride + kt].d;
                }
                أرجع;
            }
#جيڪڏهن_پورو
            uint8_t tile_quants[32][32];
            لكل (صحيح صف = 0; صف < 32; صف++) {
                ثابت int64_t r = ct * 32 + صف;
                إذا (r < ne1 && kt < ne0 / 32) {
                    ثابت بلاڪ_q4_0 & b = src_ٽڪرو[r * صف_stride + kt];
                    unpack_nibbles_32(tile_quants[صف], b.qs);
                    scale_dst[صف] = b.d;
                } وإلا {
                    memset(tile_quants[صف], 8, 32);
                    scale_dst[صف] = 0;
                }
            }
            transpose_nibbles_منزل_tile(tile_dst, tile_quants);
        });
}

ساكن فراغ repack_tiled_q4_0(فراغ * معلومات, ثابت ggml_tensor * t, size_t وٿي, size_t size) {
    unpack_from_tiled_grid_32<بلاڪ_q4_0, 1>(معلومات, t, وٿي, size, HTP_TENSOR_TILE_سائيز_Q4_0,
        [](بلاڪ_q4_0 * dst_ٽڪرو, ثابت uint8_t * tile_src, int64_t ne0, int64_t شروعات_صف, int64_t پڇاڙي_صف, int64_t صف_stride, صحيح ct, صحيح kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_منزل_nibbles(tile_quants, tile_src);
            ثابت ggml_half * scale_src = (ثابت ggml_half *) (tile_src + 512);
            لكل (صحيح صف = 0; صف < 32; صف++) {
                ثابت int64_t r = ct * 32 + صف;
                إذا (r >= شروعات_صف && r < پڇاڙي_صف && kt < ne0 / 32) {
                    بلاڪ_q4_0 & b = dst_ٽڪرو[(r - شروعات_صف) * صف_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[صف]);
                    b.d = scale_src[صف];
                }
            }
        });
}

ساكن فراغ repack_q4_1_tiled(ggml_tensor * t, ثابت فراغ * معلومات, size_t وٿي, size_t size) {
    repack_منزل_tiled_grid_32<بلاڪ_q4_1, 1>(t, معلومات, وٿي, size, HTP_TENSOR_TILE_سائيز_Q4_1,
        [](uint8_t * tile_dst, ثابت بلاڪ_q4_1 * src_ٽڪرو, int64_t ne0, int64_t ne1, int64_t صف_stride, صحيح ct, صحيح kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
#إذا defined(__ARM_NEON) || defined(__aarch64__)
            إذا (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                لكل (صحيح row0 = 0; row0 < 32; row0 += 8) {
                    ثابت uint8_t * qs[8];
                    لكل (صحيح rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_ٽڪرو[(ct * 32 + row0 + rr) * صف_stride + kt].qs;
                    }
                    repack_neon_4bit_صفون(qs, tile_dst, row0);
                }

                لكل (صحيح صف = 0; صف < 32; ++صف) {
                    ثابت بلاڪ_q4_1 & src_بلاڪ = src_ٽڪرو[(ct * 32 + صف) * صف_stride + kt];
                    scale_dst[2 * صف + 0] = src_بلاڪ.d;
                    scale_dst[2 * صف + 1] = src_بلاڪ.m;
                }
                أرجع;
            }
#جيڪڏهن_پورو
            uint8_t tile_quants[32][32];
            لكل (صحيح صف = 0; صف < 32; صف++) {
                ثابت int64_t r = ct * 32 + صف;
                إذا (r < ne1 && kt < ne0 / 32) {
                    ثابت بلاڪ_q4_1 & b = src_ٽڪرو[r * صف_stride + kt];
                    unpack_nibbles_32(tile_quants[صف], b.qs);
                    scale_dst[2 * صف + 0] = b.d;
                    scale_dst[2 * صف + 1] = b.m;
                } وإلا {
                    memset(tile_quants[صف], 0, 32);
                    scale_dst[2 * صف + 0] = 0;
                    scale_dst[2 * صف + 1] = 0;
                }
            }
            transpose_nibbles_منزل_tile(tile_dst, tile_quants);
        });
}

ساكن فراغ repack_tiled_q4_1(فراغ * معلومات, ثابت ggml_tensor * t, size_t وٿي, size_t size) {
    unpack_from_tiled_grid_32<بلاڪ_q4_1, 1>(معلومات, t, وٿي, size, HTP_TENSOR_TILE_سائيز_Q4_1,
        [](بلاڪ_q4_1 * dst_ٽڪرو, ثابت uint8_t * tile_src, int64_t ne0, int64_t شروعات_صف, int64_t پڇاڙي_صف, int64_t صف_stride, صحيح ct, صحيح kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_منزل_nibbles(tile_quants, tile_src);
            ثابت ggml_half * scale_src = (ثابت ggml_half *) (tile_src + 512);
            لكل (صحيح صف = 0; صف < 32; صف++) {
                ثابت int64_t r = ct * 32 + صف;
                إذا (r >= شروعات_صف && r < پڇاڙي_صف && kt < ne0 / 32) {
                    بلاڪ_q4_1 & b = dst_ٽڪرو[(r - شروعات_صف) * صف_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[صف]);
                    b.d = scale_src[2 * صف + 0];
                    b.m = scale_src[2 * صف + 1];
                }
            }
        });
}

ساكن فراغ repack_mxfp4_tiled(ggml_tensor * t, ثابت فراغ * معلومات, size_t وٿي, size_t size) {
    repack_منزل_tiled_grid_32<بلاڪ_mxfp4, 1>(t, معلومات, وٿي, size, HTP_TENSOR_TILE_سائيز_MXFP4,
        [](uint8_t * tile_dst, ثابت بلاڪ_mxfp4 * src_ٽڪرو, int64_t ne0, int64_t ne1, int64_t صف_stride, صحيح ct, صحيح kt) {
            uint8_t * scale_dst = tile_dst + 512;
#إذا defined(__ARM_NEON) || defined(__aarch64__)
            إذا (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                لكل (صحيح row0 = 0; row0 < 32; row0 += 8) {
                    ثابت uint8_t * qs[8];
                    لكل (صحيح rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_ٽڪرو[(ct * 32 + row0 + rr) * صف_stride + kt].qs;
                    }
                    repack_neon_4bit_صفون(qs, tile_dst, row0);
                }

                لكل (صحيح صف = 0; صف < 32; ++صف) {
                    scale_dst[صف] = src_ٽڪرو[(ct * 32 + صف) * صف_stride + kt].e;
                }
                أرجع;
            }
#جيڪڏهن_پورو
            uint8_t tile_quants[32][32];
            لكل (صحيح صف = 0; صف < 32; صف++) {
                ثابت int64_t r = ct * 32 + صف;
                إذا (r < ne1 && kt < ne0 / 32) {
                    ثابت بلاڪ_mxfp4 & b = src_ٽڪرو[r * صف_stride + kt];
                    unpack_nibbles_32(tile_quants[صف], b.qs);
                    scale_dst[صف] = b.e;
                } وإلا {
                    memset(tile_quants[صف], 0, 32);
                    scale_dst[صف] = 0;
                }
            }
            transpose_nibbles_منزل_tile(tile_dst, tile_quants);
        });
}

ساكن فراغ repack_tiled_mxfp4(فراغ * معلومات, ثابت ggml_tensor * t, size_t وٿي, size_t size) {
    unpack_from_tiled_grid_32<بلاڪ_mxfp4, 1>(معلومات, t, وٿي, size, HTP_TENSOR_TILE_سائيز_MXFP4,
        [](بلاڪ_mxfp4 * dst_ٽڪرو, ثابت uint8_t * tile_src, int64_t ne0, int64_t شروعات_صف, int64_t پڇاڙي_صف, int64_t صف_stride, صحيح ct, صحيح kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_منزل_nibbles(tile_quants, tile_src);
            ثابت uint8_t * scale_src = tile_src + 512;
            لكل (صحيح صف = 0; صف < 32; صف++) {
                ثابت int64_t r = ct * 32 + صف;
                إذا (r >= شروعات_صف && r < پڇاڙي_صف && kt < ne0 / 32) {
                    بلاڪ_mxfp4 & b = dst_ٽڪرو[(r - شروعات_صف) * صف_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[صف]);
                    b.e = scale_src[صف];
                }
            }
        });
}

ساكن فراغ repack_q8_0_tiled(ggml_tensor * t, ثابت فراغ * معلومات, size_t وٿي, size_t size) {
    repack_منزل_tiled_grid_32<بلاڪ_q8_0, 1>(t, معلومات, وٿي, size, HTP_TENSOR_TILE_سائيز_Q8_0,
        [](uint8_t * tile_dst, ثابت بلاڪ_q8_0 * src_ٽڪرو, int64_t ne0, int64_t ne1, int64_t صف_stride, صحيح ct, صحيح kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 1024);
#إذا defined(__ARM_NEON) || defined(__aarch64__)
            إذا (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                لكل (صحيح row0 = 0; row0 < 32; row0 += 8) {
                    ثابت uint8_t * qs[8];
                    لكل (صحيح rr = 0; rr < 8; ++rr) {
                        qs[rr] = (ثابت uint8_t *) src_ٽڪرو[(ct * 32 + row0 + rr) * صف_stride + kt].qs;
                    }
                    repack_neon_q8_صفون(qs, tile_dst, row0);
                }

                لكل (صحيح صف = 0; صف < 32; ++صف) {
                    scale_dst[صف] = src_ٽڪرو[(ct * 32 + صف) * صف_stride + kt].d;
                }
                أرجع;
            }
#جيڪڏهن_پورو
            لكل (صحيح cp = 0; cp < 16; cp++) {
                ثابت صحيح col0 = cp * 2;
                ثابت صحيح col1 = col0 + 1;
                لكل (صحيح صف = 0; صف < 32; صف++) {
                    ثابت int64_t r = ct * 32 + صف;
                    ثابت بلاڪ_q8_0 * b = (r < ne1 && kt < ne0 / 32) ? &src_ٽڪرو[r * صف_stride + kt] : خالي_اشارو;
                    tile_dst[cp * 64 + 2 * صف + 0] = b ? b->qs[col0] : 0;
                    tile_dst[cp * 64 + 2 * صف + 1] = b ? b->qs[col1] : 0;
                }
            }
            لكل (صحيح صف = 0; صف < 32; صف++) {
                ثابت int64_t r = ct * 32 + صف;
                scale_dst[صف] = (r < ne1 && kt < ne0 / 32) ? src_ٽڪرو[r * صف_stride + kt].d : 0;
            }
        });
}

ساكن فراغ repack_tiled_q8_0(فراغ * معلومات, ثابت ggml_tensor * t, size_t وٿي, size_t size) {
    unpack_from_tiled_grid_32<بلاڪ_q8_0, 1>(معلومات, t, وٿي, size, HTP_TENSOR_TILE_سائيز_Q8_0,
        [](بلاڪ_q8_0 * dst_ٽڪرو, ثابت uint8_t * tile_src, int64_t ne0, int64_t شروعات_صف, int64_t پڇاڙي_صف, int64_t صف_stride, صحيح ct, صحيح kt) {
            لكل (صحيح cp = 0; cp < 16; cp++) {
                ثابت صحيح col0 = cp * 2;
                ثابت صحيح col1 = col0 + 1;
                لكل (صحيح صف = 0; صف < 32; صف++) {
                    ثابت int64_t r = ct * 32 + صف;
                    إذا (r >= شروعات_صف && r < پڇاڙي_صف && kt < ne0 / 32) {
                        بلاڪ_q8_0 & b = dst_ٽڪرو[(r - شروعات_صف) * صف_stride + kt];
                        b.qs[col0] = tile_src[cp * 64 + 2 * صف + 0];
                        b.qs[col1] = tile_src[cp * 64 + 2 * صف + 1];
                    }
                }
            }
            ثابت ggml_half * scale_src = (ثابت ggml_half *) (tile_src + 1024);
            لكل (صحيح صف = 0; صف < 32; صف++) {
                ثابت int64_t r = ct * 32 + صف;
                إذا (r >= شروعات_صف && r < پڇاڙي_صف && kt < ne0 / 32) {
                    dst_ٽڪرو[(r - شروعات_صف) * صف_stride + kt].d = scale_src[صف];
                }
            }
        });
}

// --- Float Types (F16, F32, BF16 -> f16_tiled HMX layout) ---

ساكن فراغ repack_q2_0_tiled(ggml_tensor * t, ثابت فراغ * معلومات, size_t وٿي, size_t size) {
    GGML_ASSERT(وٿي == 0);
    GGML_ASSERT(t->ne[0] % 64 == 0);
    GGML_ASSERT(size == ggml_nbytes(t));

    ثابت بلاڪ_q2_0 * src = (ثابت بلاڪ_q2_0 *) معلومات;
    ثابت int64_t بلاڪون_في_صف = t->ne[0] / 64;
    ثابت int64_t n_کانو_tiles = hex_ڦيرو_مٿي(t->ne[1], 32) / 32;
    ثابت int64_t n_وزن_tiles = بلاڪون_في_صف;
    ثابت size_t ميٽرڪس_سائيز = (size_t) n_کانو_tiles * n_وزن_tiles * HTP_TENSOR_TILE_سائيز_Q2_0;

    لكل (int64_t ٽڪرو_2 = 0; ٽڪرو_2 < t->ne[2] * t->ne[3]; ++ٽڪرو_2) {
        ثابت بلاڪ_q2_0 * src_ٽڪرو = src + ٽڪرو_2 * t->ne[1] * بلاڪون_في_صف;
        uint8_t * dst_ٽڪرو = (uint8_t *) t->معلومات + ٽڪرو_2 * ميٽرڪس_سائيز;
        ثابت size_t ct_بائيٽون = (size_t) n_وزن_tiles * HTP_TENSOR_TILE_سائيز_Q2_0;

        repack_parallel_for_ct(ميٽرڪس_سائيز, (صحيح) n_کانو_tiles, [&](صحيح ct0, صحيح ct1) {
            memset(dst_ٽڪرو + (size_t) ct0 * ct_بائيٽون, 0, (size_t) (ct1 - ct0) * ct_بائيٽون);
            ثابت int64_t r0 = (int64_t) ct0 * 32;
            ثابت int64_t r1 = (std::min)((int64_t) ct1 * 32, t->ne[1]);
            لكل (int64_t صف_اشاريو = r0; صف_اشاريو < r1; ++صف_اشاريو) {
                ثابت int64_t ct = صف_اشاريو / 32;
                ثابت صحيح صف = (صحيح) (صف_اشاريو % 32);
                لكل (int64_t kt = 0; kt < n_وزن_tiles; ++kt) {
                    ثابت بلاڪ_q2_0 * بلاڪ = src_ٽڪرو + صف_اشاريو * بلاڪون_في_صف + kt;
                    uint8_t * tile = dst_ٽڪرو + (ct * n_وزن_tiles + kt) * HTP_TENSOR_TILE_سائيز_Q2_0;
#إذا defined(__ARM_NEON) || defined(__aarch64__)
                    repack_neon_q2_0_بلاڪ(بلاڪ->qs, tile, صف);
#وإلا
                    لكل (صحيح گروهه = 0; گروهه < 16; ++گروهه) {
                        لكل (صحيح byte = 0; byte < 4; ++byte) {
                            ثابت صحيح k = 4 * گروهه + byte;
                            ثابت uint8_t code = (بلاڪ->qs[k / 4] >> (2 * (k % 4))) & 3;
                            tile[(گروهه >> 2) * 128 + 4 * صف + byte] |= code << (2 * (گروهه & 3));
                        }
                    }
#جيڪڏهن_پورو
                    يادگيري_نقل(tile + 512 + 2 * صف, &بلاڪ->d, حجم(بلاڪ->d));
                }
            }
        });
    }
}

ساكن فراغ repack_tiled_q2_0(فراغ * معلومات, ثابت ggml_tensor * t, size_t وٿي, size_t size) {
    GGML_ASSERT(t->ne[0] % 64 == 0);
    GGML_ASSERT(وٿي + size <= ggml_nbytes(t));

    ثابت size_t پهريون_بلاڪ = وٿي / حجم(بلاڪ_q2_0);
    ثابت size_t پڇاڙي_بلاڪ = (وٿي + size + حجم(بلاڪ_q2_0) - 1) / حجم(بلاڪ_q2_0);
    ثابت size_t بلاڪون_في_صف = t->ne[0] / 64;
    ثابت size_t بلاڪون_في_ٽڪرو = t->ne[1] * بلاڪون_في_صف;
    ثابت size_t n_وزن_tiles = بلاڪون_في_صف;
    ثابت size_t n_کانو_tiles = hex_ڦيرو_مٿي(t->ne[1], 32) / 32;
    ثابت size_t ميٽرڪس_سائيز = n_کانو_tiles * n_وزن_tiles * HTP_TENSOR_TILE_سائيز_Q2_0;

    لكل (size_t بلاڪ_اشاريو = پهريون_بلاڪ; بلاڪ_اشاريو < پڇاڙي_بلاڪ; ++بلاڪ_اشاريو) {
        ثابت size_t ٽڪرو_2 = بلاڪ_اشاريو / بلاڪون_في_ٽڪرو;
        ثابت size_t اندريون_ٽڪرو = بلاڪ_اشاريو % بلاڪون_في_ٽڪرو;
        ثابت size_t صف_اشاريو = اندريون_ٽڪرو / بلاڪون_في_صف;
        ثابت size_t kt = اندريون_ٽڪرو % بلاڪون_في_صف;
        ثابت size_t ct = صف_اشاريو / 32;
        ثابت size_t صف = صف_اشاريو % 32;
        ثابت uint8_t * tile = (ثابت uint8_t *) t->معلومات + ٽڪرو_2 * ميٽرڪس_سائيز +
            (ct * n_وزن_tiles + kt) * HTP_TENSOR_TILE_سائيز_Q2_0;

        بلاڪ_q2_0 بلاڪ = {};
        يادگيري_نقل(&بلاڪ.d, tile + 512 + 2 * صف, حجم(بلاڪ.d));
        لكل (صحيح گروهه = 0; گروهه < 16; ++گروهه) {
            لكل (صحيح byte = 0; byte < 4; ++byte) {
                ثابت uint8_t packed = tile[(گروهه >> 2) * 128 + 4 * صف + byte];
                ثابت uint8_t code = (packed >> (2 * (گروهه & 3))) & 3;
                بلاڪ.qs[گروهه] |= code << (2 * byte);
            }
        }

        ثابت size_t بلاڪ_begin = بلاڪ_اشاريو * حجم(بلاڪ_q2_0);
        ثابت size_t copy_begin = (std::max)(بلاڪ_begin, وٿي);
        ثابت size_t copy_پڇاڙي = (std::min)(بلاڪ_begin + حجم(بلاڪ_q2_0), وٿي + size);
        يادگيري_نقل((uint8_t *) معلومات + copy_begin - وٿي, (ثابت uint8_t *) &بلاڪ + copy_begin - بلاڪ_begin, copy_پڇاڙي - copy_begin);
    }
}

سانچو <قسم_نالو SrcT, قسم_نالو منزل_Fp16_Fn>
ساكن فراغ repack_float_منزل_f16_tiled(ggml_tensor * t, ثابت فراغ * معلومات, size_t وٿي, size_t size, منزل_Fp16_Fn && منزل_fp16) {
    repack_منزل_tiled_grid_32<SrcT, 32>(t, معلومات, وٿي, size, HTP_TENSOR_TILE_سائيز_F16,
        [&](uint8_t * tile_بائيٽون, ثابت SrcT * src_ٽڪرو, int64_t ne0, int64_t ne1, int64_t صف_stride, صحيح ct, صحيح kt) {
            uint16_t * tile_dst = (uint16_t *) tile_بائيٽون;
#إذا defined(__ARM_NEON) || defined(__aarch64__)
            إذا (ct * 32 + 32 <= ne1 && kt * 32 + 32 <= ne0) {
                repack_neon_f16_tile(tile_dst, src_ٽڪرو, صف_stride, ct, kt);
                أرجع;
            }
#جيڪڏهن_پورو
            لكل (صحيح cp = 0; cp < 16; cp++) {
                ثابت int64_t c0 = kt * 32 + cp * 2;
                ثابت int64_t c1 = c0 + 1;
                لكل (صحيح صف = 0; صف < 32; صف++) {
                    ثابت int64_t r = ct * 32 + صف;
                    ثابت SrcT * r_ptr = (r < ne1) ? &src_ٽڪرو[r * صف_stride] : خالي_اشارو;
                    tile_dst[cp * 64 + صف * 2 + 0] = (r_ptr && c0 < ne0) ? منزل_fp16(r_ptr[c0]) : 0;
                    tile_dst[cp * 64 + صف * 2 + 1] = (r_ptr && c1 < ne0) ? منزل_fp16(r_ptr[c1]) : 0;
                }
            }
        });
}

سانچو <قسم_نالو DstT, قسم_نالو FromFp16Fn>
ساكن فراغ unpack_f16_tiled_منزل_float(فراغ * معلومات, ثابت ggml_tensor * t, size_t وٿي, size_t size, FromFp16Fn && from_fp16) {
    unpack_from_tiled_grid_32<DstT, 32>(معلومات, t, وٿي, size, HTP_TENSOR_TILE_سائيز_F16,
        [&](DstT * dst_ٽڪرو, ثابت uint8_t * tile_بائيٽون, int64_t ne0, int64_t شروعات_صف, int64_t پڇاڙي_صف, int64_t صف_stride, صحيح ct, صحيح kt) {
            ثابت uint16_t * tile_src = (ثابت uint16_t *) tile_بائيٽون;
            لكل (صحيح cp = 0; cp < 16; cp++) {
                ثابت int64_t c0 = kt * 32 + cp * 2;
                ثابت int64_t c1 = c0 + 1;
                لكل (صحيح صف = 0; صف < 32; صف++) {
                    ثابت int64_t r = ct * 32 + صف;
                    إذا (r >= شروعات_صف && r < پڇاڙي_صف) {
                        DstT * r_ptr = dst_ٽڪرو + (r - شروعات_صف) * صف_stride;
                        إذا (c0 < ne0) r_ptr[c0] = from_fp16(tile_src[cp * 64 + صف * 2 + 0]);
                        إذا (c1 < ne0) r_ptr[c1] = from_fp16(tile_src[cp * 64 + صف * 2 + 1]);
                    }
                }
            }
        });
}

// --- Generic K-Quant (super-block 256) grid walkers ---

سانچو <قسم_نالو بلاڪ_T, قسم_نالو PackKtFn>
ساكن فراغ repack_kquant_منزل_tiled_grid(ggml_tensor * t, ثابت فراغ * معلومات, size_t وٿي, size_t size,
                                        size_t tile_سائيز, PackKtFn && pack_kt) {
    GGML_ASSERT(وٿي == 0);
    GGML_اڻ_استعمال_ٿيل(size);

    ثابت بلاڪ_T * src_ميٽرڪس = (ثابت بلاڪ_T *) معلومات;
    ثابت int64_t ne0 = t->ne[0];
    ثابت int64_t ne1 = t->ne[1];
    ثابت int64_t ne2 = t->ne[2];
    ثابت int64_t ne3 = t->ne[3];
    GGML_ASSERT(ne0 % QK_K == 0);

    ثابت صحيح n_کانو_tiles    = hex_ڦيرو_مٿي(ne1, 32) / 32;
    ثابت صحيح n_k_tiles      = hex_ڦيرو_مٿي(ne0, 32) / 32;
    ثابت size_t ميٽرڪس_سائيز = (size_t) n_کانو_tiles * n_k_tiles * tile_سائيز;
    ثابت int64_t sb_في_صف = ne0 / QK_K;

    لكل (int64_t ٽڪرو_idx = 0; ٽڪرو_idx < ne2 * ne3; ٽڪرو_idx++) {
        ثابت بلاڪ_T * src_ٽڪرو = src_ميٽرڪس + ٽڪرو_idx * (ne1 * sb_في_صف);
        uint8_t * ميٽرڪس_dst     = (uint8_t *) t->معلومات + ٽڪرو_idx * ميٽرڪس_سائيز;
        ثابت size_t ct_بائيٽون    = (size_t) n_k_tiles * tile_سائيز;

        repack_parallel_for_ct(ميٽرڪس_سائيز, n_کانو_tiles, [&](صحيح ct0, صحيح ct1) {
            // Padding rows and OR-ed bit planes require zeroed tiles
            memset(ميٽرڪس_dst + (size_t) ct0 * ct_بائيٽون, 0, (size_t) (ct1 - ct0) * ct_بائيٽون);

            ثابت int64_t r0 = (int64_t) ct0 * 32;
            ثابت int64_t r1 = (std::min)((int64_t) ct1 * 32, ne1);
            لكل (int64_t r = r0; r < r1; r++) {
                ثابت صحيح ct  = (صحيح) (r / 32);
                ثابت صحيح صف = (صحيح) (r % 32);
                ثابت بلاڪ_T * src_صف = src_ٽڪرو + r * sb_في_صف;

                لكل (صحيح kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile = ميٽرڪس_dst + ((size_t) ct * n_k_tiles + kt) * tile_سائيز;
                    pack_kt(tile, &src_صف[kt / 8], صف, kt % 8);
                }
            }
        });
    }
}

سانچو <قسم_نالو بلاڪ_T, قسم_نالو UnpackSbFn>
ساكن فراغ unpack_kquant_from_tiled_grid(فراغ * معلومات, ثابت ggml_tensor * t, size_t وٿي, size_t size,
                                          size_t tile_سائيز, UnpackSbFn && unpack_sb) {
    uint8_t * dst_بائيٽون = (uint8_t *) معلومات;
    ثابت int64_t ne0 = t->ne[0];
    ثابت int64_t ne1 = t->ne[1];
    ثابت int64_t ne2 = t->ne[2];
    ثابت int64_t ne3 = t->ne[3];
    GGML_ASSERT(ne0 % QK_K == 0);

    ثابت صحيح n_کانو_tiles    = hex_ڦيرو_مٿي(ne1, 32) / 32;
    ثابت صحيح n_k_tiles      = hex_ڦيرو_مٿي(ne0, 32) / 32;
    ثابت size_t ميٽرڪس_سائيز = (size_t) n_کانو_tiles * n_k_tiles * tile_سائيز;
    ثابت int64_t sb_في_صف = ne0 / QK_K;

    ثابت size_t صف_سائيز_بائيٽون = ggml_صف_سائيز(t->type, ne0);
    ثابت size_t ٽڪرو_سائيز     = (size_t) ne1 * صف_سائيز_بائيٽون;
    ثابت int64_t شروعات_ٽڪرو   = وٿي / ٽڪرو_سائيز;
    int64_t پڇاڙي_ٽڪرو           = (وٿي + size + ٽڪرو_سائيز - 1) / ٽڪرو_سائيز;
    إذا (پڇاڙي_ٽڪرو > ne2 * ne3) {
        پڇاڙي_ٽڪرو = ne2 * ne3;
    }

    لكل (int64_t ٽڪرو_idx = شروعات_ٽڪرو; ٽڪرو_idx < پڇاڙي_ٽڪرو; ٽڪرو_idx++) {
        ثابت size_t cur_شروعات_byte     = (std::max)(وٿي, (size_t) ٽڪرو_idx * ٽڪرو_سائيز);
        ثابت size_t cur_پڇاڙي_byte       = (std::min)(وٿي + size, (size_t) (ٽڪرو_idx + 1) * ٽڪرو_سائيز);
        ثابت size_t ٽڪرو_وٿي_شروعات = cur_شروعات_byte - (size_t) ٽڪرو_idx * ٽڪرو_سائيز;
        ثابت size_t ٽڪرو_وٿي_پڇاڙي   = cur_پڇاڙي_byte - (size_t) ٽڪرو_idx * ٽڪرو_سائيز;

        ثابت int64_t شروعات_صف = ٽڪرو_وٿي_شروعات / صف_سائيز_بائيٽون;
        ثابت int64_t پڇاڙي_صف   = (std::min)((int64_t) ((ٽڪرو_وٿي_پڇاڙي + صف_سائيز_بائيٽون - 1) / صف_سائيز_بائيٽون), ne1);

        بلاڪ_T * dst_ٽڪرو         = (بلاڪ_T *) (dst_بائيٽون + (cur_شروعات_byte - وٿي));
        ثابت uint8_t * ميٽرڪس_src = (ثابت uint8_t *) t->معلومات + ٽڪرو_idx * ميٽرڪس_سائيز;

        لكل (int64_t r = شروعات_صف; r < پڇاڙي_صف; r++) {
            ثابت صحيح ct  = (صحيح) (r / 32);
            ثابت صحيح صف = (صحيح) (r % 32);
            بلاڪ_T * dst_صف = dst_ٽڪرو + (r - شروعات_صف) * sb_في_صف;

            لكل (int64_t sb = 0; sb < sb_في_صف; sb++) {
                بلاڪ_T * b = &dst_صف[sb];
                memset(b, 0, حجم(بلاڪ_T));
                ثابت uint8_t * sb_tiles = ميٽرڪس_src + ((size_t) ct * n_k_tiles + sb * 8) * tile_سائيز;
                unpack_sb(b, sb_tiles, tile_سائيز, صف);
            }
        }
    }
}

// --- K-Quant Helpers & Per-Type Implementations (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) ---

#إذا !defined(__ARM_NEON) && !defined(__aarch64__)
// unsigned 6-bit value (0..63) of element e of a Q6_K block, same bit layout as dequantize_row_q6_K
ساكن مضمن uint8_t q6_K_get_quant(ثابت بلاڪ_q6_K * b, صحيح e) {
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
#جيڪڏهن_پورو

ساكن فراغ repack_q6_K_tiled(ggml_tensor * t, ثابت فراغ * معلومات, size_t وٿي, size_t size) {
    repack_kquant_منزل_tiled_grid<بلاڪ_q6_K>(t, معلومات, وٿي, size, HTP_TENSOR_TILE_سائيز_Q6_K,
        [](uint8_t * tile, ثابت بلاڪ_q6_K * b, صحيح صف, صحيح kt_local) {
            ثابت عائم d     = GGML_FP16_منزل_FP32(b->d);
            uint8_t * lo_pl   = tile;
            uint8_t * hi_pl   = tile + 512;
            ggml_half * sc_pl = (ggml_half *) (tile + 768);

#إذا defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q6_k(b, kt_local, lo_pl, hi_pl, صف);
#وإلا
            لكل (صحيح lk = 0; lk < 32; lk++) {
                ثابت uint8_t q6 = q6_K_get_quant(b, kt_local * 32 + lk);
                ثابت صحيح g   = lk >> 2;
                ثابت صحيح pos = صف * 4 + (lk & 3);
                lo_pl[(g >> 1) * 128 + pos] |= (uint8_t) ((q6 & 0xF) << ((g & 1) * 4));
                hi_pl[(g >> 2) * 128 + pos] |= (uint8_t) ((q6 >> 4) << ((g & 3) * 2));
            }
#جيڪڏهن_پورو
            لكل (صحيح sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + صف] = GGML_FP32_منزل_FP16(d * (عائم) b->scales[kt_local * 2 + sub]);
            }
        });
}

ساكن فراغ repack_tiled_q6_K(فراغ * معلومات, ثابت ggml_tensor * t, size_t وٿي, size_t size) {
    unpack_kquant_from_tiled_grid<بلاڪ_q6_K>(معلومات, t, وٿي, size, HTP_TENSOR_TILE_سائيز_Q6_K,
        [](بلاڪ_q6_K * b, ثابت uint8_t * sb_tiles, size_t tile_سائيز, صحيح صف) {
            عائم sub_scales[16];
            لكل (صحيح kt_local = 0; kt_local < 8; kt_local++) {
                ثابت uint8_t *   tile  = sb_tiles + kt_local * tile_سائيز;
                ثابت uint8_t *   lo_pl = tile;
                ثابت uint8_t *   hi_pl = tile + 512;
                ثابت ggml_half * sc_pl = (ثابت ggml_half *) (tile + 768);

                ثابت صحيح c  = kt_local / 4;
                ثابت صحيح g  = kt_local % 4;
                uint8_t * ql = b->ql + c * 64;
                uint8_t * qh = b->qh + c * 32;

                لكل (صحيح lk = 0; lk < 32; lk++) {
                    ثابت صحيح g_tile = lk >> 2;
                    ثابت صحيح pos    = صف * 4 + (lk & 3);
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
                    sub_scales[kt_local * 2 + sub] = GGML_FP16_منزل_FP32(sc_pl[sub * 32 + صف]);
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
ساكن مضمن منطقي q3_K_get_hbit(ثابت بلاڪ_q3_K * b, صحيح e) {
    ثابت صحيح c = e / 128;
    ثابت صحيح j = (e % 128) / 32;
    ثابت صحيح l = e % 32;
    أرجع (b->hmask[l] >> (c * 4 + j)) & 1;
}
#جيڪڏهن_پورو

// signed 6-bit scale j (-32..31) of a Q3_K block, same packing as quantize_row_q3_K_ref
ساكن مضمن صحيح q3_K_get_scale(ثابت uint8_t * scales, صحيح j) {
    ثابت صحيح lo = (j < 8) ? (scales[j] & 0xF) : (scales[j - 8] >> 4);
    ثابت صحيح hi = (scales[8 + j % 4] >> (2 * (j / 4))) & 3;
    أرجع (lo | (hi << 4)) - 32;
}

// read-back: find fp16 d and l[j] in [lmin, lmax] with fp16(d * l[j]) == prod[j] for all j, false if none
ساكن منطقي hexagon_recover_k_scales(ثابت ggml_half * prod, صحيح n, صحيح lmin, صحيح lmax, ggml_half * d_ٻاهريون, صحيح * l_ٻاهريون) {
    صحيح jmax = 0;
    لكل (صحيح j = 1; j < n; j++) {
        إذا (fabsf(GGML_FP16_منزل_FP32(prod[j])) > fabsf(GGML_FP16_منزل_FP32(prod[jmax]))) {
            jmax = j;
        }
    }
    ثابت عائم pmax = GGML_FP16_منزل_FP32(prod[jmax]);
    إذا (pmax == 0.0f) {
        *d_ٻاهريون = GGML_FP32_منزل_FP16(0.0f);
        لكل (صحيح j = 0; j < n; j++) {
            l_ٻاهريون[j] = 0;
        }
        أرجع سچ;
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
                uint16_t بٽ;
                يادگيري_نقل(&بٽ, &d, حجم(بٽ));
                بٽ = (uint16_t) (بٽ + ulp);
                يادگيري_نقل(&d, &بٽ, حجم(بٽ));

                ثابت عائم df = GGML_FP16_منزل_FP32(d);
                إذا (!std::isfinite(df) || df == 0.0f) {
                    استمر;
                }
                منطقي ok = سچ;
                لكل (صحيح j = 0; j < n && ok; j++) {
                    ثابت صحيح l = (صحيح) roundf(GGML_FP16_منزل_FP32(prod[j]) / df);
                    ثابت ggml_half p = GGML_FP32_منزل_FP16(df * (عائم) l);
                    ok = l >= lmin && l <= lmax && memcmp(&p, &prod[j], حجم(p)) == 0;
                    l_ٻاهريون[j] = l;
                }
                إذا (ok) {
                    *d_ٻاهريون = d;
                    أرجع سچ;
                }
            }
        }
    }
    أرجع خطأ;
}

ساكن فراغ repack_q3_K_tiled(ggml_tensor * t, ثابت فراغ * معلومات, size_t وٿي, size_t size) {
    repack_kquant_منزل_tiled_grid<بلاڪ_q3_K>(t, معلومات, وٿي, size, HTP_TENSOR_TILE_سائيز_Q3_K,
        [](uint8_t * tile, ثابت بلاڪ_q3_K * b, صحيح صف, صحيح kt_local) {
            ثابت عائم d     = GGML_FP16_منزل_FP32(b->d);
            uint8_t * lo_pl   = tile;
            uint8_t * neg_pl  = tile + 256;
            ggml_half * sc_pl = (ggml_half *) (tile + 384);

#إذا defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q2_3_k_low2(b->qs, kt_local, lo_pl, صف);
            repack_neon_q3_k_hmask(b->hmask, kt_local, neg_pl, صف);
#وإلا
            لكل (صحيح lk = 0; lk < 32; lk++) {
                ثابت صحيح e   = kt_local * 32 + lk;
                ثابت صحيح g   = lk >> 2;
                ثابت صحيح pos = صف * 4 + (lk & 3);
                lo_pl[(g >> 2) * 128 + pos] |= (uint8_t) (q2_3_K_get_low2(b->qs, e) << ((g & 3) * 2));
                إذا (!q3_K_get_hbit(b, e)) {
                    neg_pl[pos] |= (uint8_t) (1 << g);
                }
            }
#جيڪڏهن_پورو
            لكل (صحيح sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + صف] = GGML_FP32_منزل_FP16(d * (عائم) q3_K_get_scale(b->scales, kt_local * 2 + sub));
            }
        });
}

ساكن فراغ repack_tiled_q3_K(فراغ * معلومات, ثابت ggml_tensor * t, size_t وٿي, size_t size) {
    unpack_kquant_from_tiled_grid<بلاڪ_q3_K>(معلومات, t, وٿي, size, HTP_TENSOR_TILE_سائيز_Q3_K,
        [](بلاڪ_q3_K * b, ثابت uint8_t * sb_tiles, size_t tile_سائيز, صحيح صف) {
            ggml_half sub_scales[16];
            لكل (صحيح kt_local = 0; kt_local < 8; kt_local++) {
                ثابت uint8_t *   tile   = sb_tiles + kt_local * tile_سائيز;
                ثابت uint8_t *   lo_pl  = tile;
                ثابت uint8_t *   neg_pl = tile + 256;
                ثابت ggml_half * sc_pl  = (ثابت ggml_half *) (tile + 384);

                لكل (صحيح lk = 0; lk < 32; lk++) {
                    ثابت صحيح e   = kt_local * 32 + lk;
                    ثابت صحيح g   = lk >> 2;
                    ثابت صحيح pos = صف * 4 + (lk & 3);
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
                    sub_scales[kt_local * 2 + sub] = sc_pl[sub * 32 + صف];
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

ساكن فراغ repack_q2_K_tiled(ggml_tensor * t, ثابت فراغ * معلومات, size_t وٿي, size_t size) {
    repack_kquant_منزل_tiled_grid<بلاڪ_q2_K>(t, معلومات, وٿي, size, HTP_TENSOR_TILE_سائيز_Q2_K,
        [](uint8_t * tile, ثابت بلاڪ_q2_K * b, صحيح صف, صحيح kt_local) {
            uint8_t * lo_pl   = tile;
            uint8_t * sc_pl   = tile + 256;
            ggml_half * dd_pl = (ggml_half *) (tile + 320);

#إذا defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q2_3_k_low2(b->qs, kt_local, lo_pl, صف);
#وإلا
            لكل (صحيح lk = 0; lk < 32; lk++) {
                ثابت صحيح g   = lk >> 2;
                ثابت صحيح pos = صف * 4 + (lk & 3);
                lo_pl[(g >> 2) * 128 + pos] |= (uint8_t) (q2_3_K_get_low2(b->qs, kt_local * 32 + lk) << ((g & 3) * 2));
            }
#جيڪڏهن_پورو
            لكل (صحيح sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + صف] = b->scales[kt_local * 2 + sub];
            }
            إذا (صف / 8 == kt_local % 4) {
                dd_pl[صف % 8] = kt_local < 4 ? b->d : (ggml_half) (b->dmin ^ 0x8000);
            }
        });
}

ساكن فراغ repack_tiled_q2_K(فراغ * معلومات, ثابت ggml_tensor * t, size_t وٿي, size_t size) {
    unpack_kquant_from_tiled_grid<بلاڪ_q2_K>(معلومات, t, وٿي, size, HTP_TENSOR_TILE_سائيز_Q2_K,
        [](بلاڪ_q2_K * b, ثابت uint8_t * sb_tiles, size_t tile_سائيز, صحيح صف) {
            لكل (صحيح kt_local = 0; kt_local < 8; kt_local++) {
                ثابت uint8_t *   tile  = sb_tiles + kt_local * tile_سائيز;
                ثابت uint8_t *   lo_pl = tile;
                ثابت uint8_t *   sc_pl = tile + 256;
                ثابت ggml_half * dd_pl = (ثابت ggml_half *) (tile + 320);

                لكل (صحيح lk = 0; lk < 32; lk++) {
                    ثابت صحيح e   = kt_local * 32 + lk;
                    ثابت صحيح g   = lk >> 2;
                    ثابت صحيح pos = صف * 4 + (lk & 3);
                    ثابت uint8_t lo = (lo_pl[(g >> 2) * 128 + pos] >> ((g & 3) * 2)) & 3;
                    b->qs[(e / 128) * 32 + e % 32] |= (uint8_t) (lo << (2 * ((e % 128) / 32)));
                }

                لكل (صحيح sub = 0; sub < 2; sub++) {
                    b->scales[kt_local * 2 + sub] = sc_pl[sub * 32 + صف];
                }
                إذا (صف / 8 == kt_local % 4) {
                    إذا (kt_local < 4) {
                        b->d = dd_pl[صف % 8];
                    } وإلا {
                        b->dmin = (ggml_half) (dd_pl[صف % 8] ^ 0x8000);
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

سانچو <قسم_نالو بلاڪ_T>
ساكن مضمن فراغ repack_k4_quants(uint8_t * tile_dst, ثابت بلاڪ_T * b, صحيح صف, صحيح kt_local,
                                    int64_t sb_في_صف, منطقي use_neon) {
#إذا defined(__ARM_NEON) || defined(__aarch64__)
    إذا (use_neon) {
        إذا ((صف & 7) == 0) {
            ثابت uint8_t * qs[8];
            لكل (صحيح rr = 0; rr < 8; ++rr) {
                qs[rr] = b[rr * sb_في_صف].qs + (kt_local / 2) * 32;
            }
            repack_neon_k4_صفون(qs, tile_dst, صف, (kt_local & 1) != 0);
        }
        أرجع;
    }
#وإلا
    GGML_اڻ_استعمال_ٿيل(sb_في_صف);
    GGML_اڻ_استعمال_ٿيل(use_neon);
#جيڪڏهن_پورو
    ثابت uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
    ثابت صحيح سرڪايو = (kt_local & 1) ? 4 : 0;

    لكل (صحيح cp = 0; cp < 16; cp++) {
        ثابت uint8_t q0 = (qs_sub[2 * cp + 0] >> سرڪايو) & 0x0F;
        ثابت uint8_t q1 = (qs_sub[2 * cp + 1] >> سرڪايو) & 0x0F;
        tile_dst[cp * 32 + صف] = (uint8_t) ((q1 << 4) | q0);
    }
}

ساكن فراغ repack_q4_K_tiled(ggml_tensor * t, ثابت فراغ * معلومات, size_t وٿي, size_t size) {
    ثابت int64_t sb_في_صف = t->ne[0] / QK_K;
    ثابت منطقي    use_neon   = (t->ne[1] % 8 == 0);
    repack_kquant_منزل_tiled_grid<بلاڪ_q4_K>(t, معلومات, وٿي, size, HTP_TENSOR_TILE_سائيز_Q4_K,
        [sb_في_صف, use_neon](uint8_t * tile_dst, ثابت بلاڪ_q4_K * b, صحيح صف, صحيح kt_local) {
            ثابت عائم d    = GGML_FP16_منزل_FP32(b->d);
            ثابت عائم dmin = GGML_FP16_منزل_FP32(b->dmin);

            uint8_t sc, m;
            get_scale_min_k4(kt_local, b->scales, &sc, &m);

            repack_k4_quants(tile_dst, b, صف, kt_local, sb_في_صف, use_neon);

            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
            scale_dst[2 * صف + 0] = GGML_FP32_منزل_FP16(d * (عائم) sc);
            scale_dst[2 * صف + 1] = GGML_FP32_منزل_FP16(-dmin * (عائم) m);
        });
}

ساكن فراغ repack_tiled_q4_K(فراغ * معلومات, ثابت ggml_tensor * t, size_t وٿي, size_t size) {
    unpack_kquant_from_tiled_grid<بلاڪ_q4_K>(معلومات, t, وٿي, size, HTP_TENSOR_TILE_سائيز_Q4_K,
        [](بلاڪ_q4_K * b, ثابت uint8_t * sb_tiles, size_t tile_سائيز, صحيح صف) {
            عائم sub_scales[8];
            عائم sub_mins[8];

            لكل (صحيح kt_local = 0; kt_local < 8; kt_local++) {
                ثابت uint8_t *   tile_src  = sb_tiles + kt_local * tile_سائيز;
                ثابت ggml_half * scale_src = (ثابت ggml_half *) (tile_src + 512);

                uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
                ثابت صحيح سرڪايو  = (kt_local & 1) ? 4 : 0;

                لكل (صحيح cp = 0; cp < 16; cp++) {
                    ثابت uint8_t val = tile_src[cp * 32 + صف];
                    qs_sub[2 * cp + 0] |= (uint8_t) ((val & 0x0F) << سرڪايو);
                    qs_sub[2 * cp + 1] |= (uint8_t) ((val >> 4)   << سرڪايو);
                }

                ثابت عائم D = GGML_FP16_منزل_FP32(scale_src[2 * صف + 0]);
                ثابت عائم M = GGML_FP16_منزل_FP32(scale_src[2 * صف + 1]);
                sub_scales[kt_local] = (D > 0.0f) ? D : 0.0f;
                sub_mins[kt_local]   = (-M > 0.0f) ? -M : 0.0f;
            }

            pack_k4_scales_mins(b->scales, &b->d, &b->dmin, sub_scales, sub_mins);
        });
}

ساكن فراغ repack_q5_K_tiled(ggml_tensor * t, ثابت فراغ * معلومات, size_t وٿي, size_t size) {
    ثابت int64_t sb_في_صف = t->ne[0] / QK_K;
    ثابت منطقي    use_neon   = (t->ne[1] % 8 == 0);
    repack_kquant_منزل_tiled_grid<بلاڪ_q5_K>(t, معلومات, وٿي, size, HTP_TENSOR_TILE_سائيز_Q5_K,
        [sb_في_صف, use_neon](uint8_t * tile_dst, ثابت بلاڪ_q5_K * b, صحيح صف, صحيح kt_local) {
            ثابت عائم d     = GGML_FP16_منزل_FP32(b->d);
            ثابت عائم dmin  = GGML_FP16_منزل_FP32(b->dmin);
            uint8_t * plane   = tile_dst + 640;

            uint8_t sc, m;
            get_scale_min_k4(kt_local, b->scales, &sc, &m);

            repack_k4_quants(tile_dst, b, صف, kt_local, sb_في_صف, use_neon);

            ثابت uint8_t hbit = (uint8_t) (1 << kt_local);
#إذا defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q5_k_qh(b->qh, hbit, plane, صف);
#وإلا
            لكل (صحيح cp = 0; cp < 16; cp++) {
                ثابت صحيح i    = cp / 4;
                ثابت صحيح lane = (cp % 4) * 32 + صف;
                إذا (b->qh[2 * cp + 0] & hbit) {
                    plane[lane] |= (uint8_t) (1 << (2 * i));
                }
                إذا (b->qh[2 * cp + 1] & hbit) {
                    plane[lane] |= (uint8_t) (1 << (2 * i + 1));
                }
            }
#جيڪڏهن_پورو

            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
            scale_dst[2 * صف + 0] = GGML_FP32_منزل_FP16(d * (عائم) sc);
            scale_dst[2 * صف + 1] = GGML_FP32_منزل_FP16(-dmin * (عائم) m);
        });
}

ساكن فراغ repack_tiled_q5_K(فراغ * معلومات, ثابت ggml_tensor * t, size_t وٿي, size_t size) {
    unpack_kquant_from_tiled_grid<بلاڪ_q5_K>(معلومات, t, وٿي, size, HTP_TENSOR_TILE_سائيز_Q5_K,
        [](بلاڪ_q5_K * b, ثابت uint8_t * sb_tiles, size_t tile_سائيز, صحيح صف) {
            عائم sub_scales[8];
            عائم sub_mins[8];

            لكل (صحيح kt_local = 0; kt_local < 8; kt_local++) {
                ثابت uint8_t *   tile_src  = sb_tiles + kt_local * tile_سائيز;
                ثابت uint8_t *   plane     = tile_src + 640;
                ثابت ggml_half * scale_src = (ثابت ggml_half *) (tile_src + 512);

                uint8_t * qs_sub   = b->qs + (kt_local / 2) * 32;
                ثابت صحيح سرڪايو    = (kt_local & 1) ? 4 : 0;
                ثابت uint8_t hbit = (uint8_t) (1 << kt_local);

                لكل (صحيح cp = 0; cp < 16; cp++) {
                    ثابت uint8_t val = tile_src[cp * 32 + صف];
                    qs_sub[2 * cp + 0] |= (uint8_t) ((val & 0x0F) << سرڪايو);
                    qs_sub[2 * cp + 1] |= (uint8_t) ((val >> 4)   << سرڪايو);

                    ثابت صحيح i    = cp / 4;
                    ثابت صحيح lane = (cp % 4) * 32 + صف;
                    إذا (plane[lane] & (1 << (2 * i))) {
                        b->qh[2 * cp + 0] |= hbit;
                    }
                    إذا (plane[lane] & (1 << (2 * i + 1))) {
                        b->qh[2 * cp + 1] |= hbit;
                    }
                }

                ثابت عائم D = GGML_FP16_منزل_FP32(scale_src[2 * صف + 0]);
                ثابت عائم M = GGML_FP16_منزل_FP32(scale_src[2 * صف + 1]);
                sub_scales[kt_local] = (D > 0.0f) ? D : 0.0f;
                sub_mins[kt_local]   = (-M > 0.0f) ? -M : 0.0f;
            }

            pack_k4_scales_mins(b->scales, &b->d, &b->dmin, sub_scales, sub_mins);
        });
}

ساكن مضمن فراغ q1_0_tiled_ٻٽو_انگ_location(صحيح k, صحيح صف, صحيح & byte_وٿي, صحيح & ٻٽو_انگ) {
    ثابت صحيح quarter = k / 32;
    ثابت صحيح kq      = k % 32;

    byte_وٿي = quarter * 128 + 4 * صف + (kq % 4);
    ٻٽو_انگ         = kq / 4;
}

ساكن فراغ repack_q1_0_tiled(ggml_tensor * t, ثابت فراغ * معلومات, size_t وٿي, size_t size) {
    ثابت بلاڪ_q1_0 * src_ميٽرڪس = (ثابت بلاڪ_q1_0 *) معلومات;
    ثابت int64_t ne0 = t->ne[0];
    ثابت int64_t ne1 = t->ne[1];
    ثابت int64_t ne2 = t->ne[2];
    ثابت int64_t ne3 = t->ne[3];
    ثابت صحيح n_کانو_tiles = hex_ڦيرو_مٿي(ne1, 32) / 32;
    ثابت صحيح n_k_tiles = ne0 / QK1_0;
    ثابت size_t tile_سائيز = HTP_TENSOR_TILE_سائيز_Q1_0;
    ثابت size_t ميٽرڪس_سائيز = (size_t) n_کانو_tiles * n_k_tiles * tile_سائيز;
    ثابت size_t ٽڪرو_سائيز = ne1 * ggml_صف_سائيز(t->type, ne0);
    ثابت int64_t شروعات_ٽڪرو = وٿي / ٽڪرو_سائيز;
    ثابت int64_t پڇاڙي_ٽڪرو = (std::min)((int64_t) ((وٿي + size + ٽڪرو_سائيز - 1) / ٽڪرو_سائيز), ne2 * ne3);

    GGML_ASSERT(ne0 % QK1_0 == 0);

    لكل (int64_t ٽڪرو_idx = شروعات_ٽڪرو; ٽڪرو_idx < پڇاڙي_ٽڪرو; ٽڪرو_idx++) {
        ثابت بلاڪ_q1_0 * src_ٽڪرو = src_ميٽرڪس + (ٽڪرو_idx - شروعات_ٽڪرو) * ne1 * n_k_tiles;
        uint8_t * ميٽرڪس_dst = (uint8_t *) t->معلومات + ٽڪرو_idx * ميٽرڪس_سائيز;

        repack_parallel_for_ct(ميٽرڪس_سائيز, n_کانو_tiles, [&](صحيح ct0, صحيح ct1) {
            لكل (صحيح ct = ct0; ct < ct1; ct++) {
                لكل (صحيح kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = ميٽرڪس_dst + ((size_t) ct * n_k_tiles + kt) * tile_سائيز;
                    ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
                    memset(tile_dst, 0, tile_سائيز);

                    لكل (صحيح صف = 0; صف < 32; صف++) {
                        ثابت int64_t r = ct * 32 + صف;
                        إذا (r >= ne1) {
                            استمر;
                        }

                        ثابت بلاڪ_q1_0 & بلاڪ = src_ٽڪرو[r * n_k_tiles + kt];
                        scale_dst[صف] = بلاڪ.d;
#إذا defined(__ARM_NEON) || defined(__aarch64__)
                        repack_neon_q1_0_بلاڪ(بلاڪ.qs, tile_dst, صف);
#وإلا
                        لكل (صحيح k = 0; k < QK1_0; k++) {
                            صحيح byte_وٿي;
                            صحيح ٻٽو_انگ;
                            q1_0_tiled_ٻٽو_انگ_location(k, صف, byte_وٿي, ٻٽو_انگ);
                            إذا (بلاڪ.qs[k / 8] & (1u << (k % 8))) {
                                tile_dst[byte_وٿي] |= (uint8_t) (1u << ٻٽو_انگ);
                            }
                        }
#جيڪڏهن_پورو
                    }
                }
            }
        });
    }
}

ساكن فراغ repack_tiled_q1_0(فراغ * معلومات, ثابت ggml_tensor * t, size_t وٿي, size_t size) {
    بلاڪ_q1_0 * dst_ميٽرڪس = (بلاڪ_q1_0 *) معلومات;
    ثابت int64_t ne0 = t->ne[0];
    ثابت int64_t ne1 = t->ne[1];
    ثابت int64_t ne2 = t->ne[2];
    ثابت int64_t ne3 = t->ne[3];
    ثابت صحيح n_کانو_tiles = hex_ڦيرو_مٿي(ne1, 32) / 32;
    ثابت صحيح n_k_tiles = ne0 / QK1_0;
    ثابت size_t tile_سائيز = HTP_TENSOR_TILE_سائيز_Q1_0;
    ثابت size_t ميٽرڪس_سائيز = (size_t) n_کانو_tiles * n_k_tiles * tile_سائيز;
    ثابت size_t صف_سائيز_بائيٽون = ggml_صف_سائيز(t->type, ne0);
    ثابت size_t ٽڪرو_سائيز = ne1 * صف_سائيز_بائيٽون;
    ثابت int64_t شروعات_ٽڪرو = وٿي / ٽڪرو_سائيز;
    ثابت int64_t پڇاڙي_ٽڪرو = (std::min)((int64_t) ((وٿي + size + ٽڪرو_سائيز - 1) / ٽڪرو_سائيز), ne2 * ne3);

    GGML_ASSERT(ne0 % QK1_0 == 0);

    لكل (int64_t ٽڪرو_idx = شروعات_ٽڪرو; ٽڪرو_idx < پڇاڙي_ٽڪرو; ٽڪرو_idx++) {
        ثابت size_t cur_شروعات_byte = (std::max)(وٿي, (size_t) ٽڪرو_idx * ٽڪرو_سائيز);
        ثابت size_t cur_پڇاڙي_byte = (std::min)(وٿي + size, (size_t) (ٽڪرو_idx + 1) * ٽڪرو_سائيز);
        ثابت int64_t شروعات_صف = (cur_شروعات_byte - (size_t) ٽڪرو_idx * ٽڪرو_سائيز) / صف_سائيز_بائيٽون;
        ثابت int64_t پڇاڙي_صف = (std::min)((int64_t) ((cur_پڇاڙي_byte - (size_t) ٽڪرو_idx * ٽڪرو_سائيز + صف_سائيز_بائيٽون - 1) / صف_سائيز_بائيٽون), ne1);
        بلاڪ_q1_0 * dst_ٽڪرو = dst_ميٽرڪس + (cur_شروعات_byte - وٿي) / حجم(بلاڪ_q1_0);
        ثابت uint8_t * ميٽرڪس_src = (ثابت uint8_t *) t->معلومات + ٽڪرو_idx * ميٽرڪس_سائيز;

        لكل (صحيح ct = شروعات_صف / 32; ct < (پڇاڙي_صف + 31) / 32; ct++) {
            لكل (صحيح kt = 0; kt < n_k_tiles; kt++) {
                ثابت uint8_t * tile_src = ميٽرڪس_src + ((size_t) ct * n_k_tiles + kt) * tile_سائيز;
                ثابت ggml_half * scale_src = (ثابت ggml_half *) (tile_src + 512);

                لكل (صحيح صف = 0; صف < 32; صف++) {
                    ثابت int64_t r = ct * 32 + صف;
                    إذا (r < شروعات_صف || r >= پڇاڙي_صف) {
                        استمر;
                    }

                    بلاڪ_q1_0 & بلاڪ = dst_ٽڪرو[(r - شروعات_صف) * n_k_tiles + kt];
                    memset(بلاڪ.qs, 0, حجم(بلاڪ.qs));
                    بلاڪ.d = scale_src[صف];
                    لكل (صحيح k = 0; k < QK1_0; k++) {
                        صحيح byte_وٿي;
                        صحيح ٻٽو_انگ;
                        q1_0_tiled_ٻٽو_انگ_location(k, صف, byte_وٿي, ٻٽو_انگ);
                        إذا (tile_src[byte_وٿي] & (1u << ٻٽو_انگ)) {
                            بلاڪ.qs[k / 8] |= (uint8_t) (1u << (k % 8));
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

فراغ htp_unpack_tensor_tiled(فراغ * معلومات, ثابت ggml_tensor * tensor, size_t وٿي, size_t size) {
    GGML_ASSERT(وٿي + size <= ggml_nbytes(tensor));

    بدل (tensor->type) {
        حالة GGML_TYPE_Q1_0:
            repack_tiled_q1_0(معلومات, tensor, وٿي, size);
            اكسر;
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_IQ4_NL:
            repack_tiled_q4_0(معلومات, tensor, وٿي, size);
            اكسر;
        حالة GGML_TYPE_Q4_1:
            repack_tiled_q4_1(معلومات, tensor, وٿي, size);
            اكسر;
        حالة GGML_TYPE_Q8_0:
            repack_tiled_q8_0(معلومات, tensor, وٿي, size);
            اكسر;
        حالة GGML_TYPE_MXFP4:
            repack_tiled_mxfp4(معلومات, tensor, وٿي, size);
            اكسر;
        حالة GGML_TYPE_Q2_0:
            repack_tiled_q2_0(معلومات, tensor, وٿي, size);
            اكسر;
        حالة GGML_TYPE_Q2_K:
            repack_tiled_q2_K(معلومات, tensor, وٿي, size);
            اكسر;
        حالة GGML_TYPE_Q3_K:
            repack_tiled_q3_K(معلومات, tensor, وٿي, size);
            اكسر;
        حالة GGML_TYPE_Q4_K:
            repack_tiled_q4_K(معلومات, tensor, وٿي, size);
            اكسر;
        حالة GGML_TYPE_Q5_K:
            repack_tiled_q5_K(معلومات, tensor, وٿي, size);
            اكسر;
        حالة GGML_TYPE_Q6_K:
            repack_tiled_q6_K(معلومات, tensor, وٿي, size);
            اكسر;
        حالة GGML_TYPE_F16:
            unpack_f16_tiled_منزل_float<uint16_t>(معلومات, tensor, وٿي, size,
                [](uint16_t v) { أرجع v; });
            اكسر;
        حالة GGML_TYPE_F32:
            unpack_f16_tiled_منزل_float<عائم>(معلومات, tensor, وٿي, size,
                [](uint16_t v) { أرجع ggml_fp16_منزل_fp32((ggml_fp16_t) v); });
            اكسر;
        حالة GGML_TYPE_BF16:
            unpack_f16_tiled_منزل_float<ggml_bf16_t>(معلومات, tensor, وٿي, size,
                [](uint16_t v) { أرجع ggml_fp32_منزل_bf16(ggml_fp16_منزل_fp32((ggml_fp16_t) v)); });
            اكسر;
        افتراضي:
            يادگيري_نقل(معلومات, (ثابت حرف *) tensor->معلومات + وٿي, size);
            اكسر;
    }
}
