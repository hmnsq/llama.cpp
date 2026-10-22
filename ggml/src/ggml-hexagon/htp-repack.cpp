#अन्तर्भाव "htp-repack.h"

#define GGML_COMMON_IMPL_CPP
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-quants.h"

#अन्तर्भाव <algorithm>
#अन्तर्भाव <cmath>
#अन्तर्भाव <cstring>
#अन्तर्भाव <initializer_list>
#अन्तर्भाव <thread>

#जर defined(__ARM_NEON) || defined(__aarch64__)
#    include <arm_neon.h>
#endif

// --- 4-bit block-32 nibble helpers (shared by Q4_0, IQ4_NL, Q4_1, MXFP4) ---

स्थिर ओळीत शून्य unpack_nibbles_32(uint8_t * dst, स्थिरांक uint8_t * qs) {
    साठी (अचिन्हित पूर्णांक i = 0; i < 16; ++i) {
        dst[i + 0]  = qs[i] & 0x0F;
        dst[i + 16] = qs[i] >> 4;
    }
}

स्थिर ओळीत शून्य pack_nibbles_32(uint8_t * qs, स्थिरांक uint8_t * src) {
    साठी (अचिन्हित पूर्णांक i = 0; i < 16; ++i) {
        qs[i] = src[i + 0] | (src[i + 16] << 4);
    }
}

स्थिर ओळीत शून्य transpose_nibbles_to_tile(uint8_t * tile_dst, स्थिरांक uint8_t tile_quants[32][32]) {
    साठी (पूर्णांक cp = 0; cp < 16; cp++) {
        साठी (पूर्णांक row = 0; row < 32; row++) {
            tile_dst[cp * 32 + row] = (tile_quants[row][2 * cp + 1] << 4) | tile_quants[row][2 * cp];
        }
    }
}

स्थिर ओळीत शून्य transpose_tile_to_nibbles(uint8_t tile_quants[32][32], स्थिरांक uint8_t * tile_src) {
    साठी (पूर्णांक cp = 0; cp < 16; cp++) {
        साठी (पूर्णांक row = 0; row < 32; row++) {
            स्थिरांक uint8_t val = tile_src[cp * 32 + row];
            tile_quants[row][2 * cp + 0] = val & 0x0F;
            tile_quants[row][2 * cp + 1] = val >> 4;
        }
    }
}

#जर defined(__ARM_NEON) || defined(__aarch64__)
स्थिर शून्य repack_neon_transpose8x8(स्थिरांक uint8x8_t rows[8], uint8x8_t columns[8]) {
    स्थिरांक uint8x8x2_t t0 = vtrn_u8(rows[0], rows[1]);
    स्थिरांक uint8x8x2_t t1 = vtrn_u8(rows[2], rows[3]);
    स्थिरांक uint8x8x2_t t2 = vtrn_u8(rows[4], rows[5]);
    स्थिरांक uint8x8x2_t t3 = vtrn_u8(rows[6], rows[7]);
    स्थिरांक uint16x4x2_t t4 = vtrn_u16(vreinterpret_u16_u8(t0.val[0]), vreinterpret_u16_u8(t1.val[0]));
    स्थिरांक uint16x4x2_t t5 = vtrn_u16(vreinterpret_u16_u8(t0.val[1]), vreinterpret_u16_u8(t1.val[1]));
    स्थिरांक uint16x4x2_t t6 = vtrn_u16(vreinterpret_u16_u8(t2.val[0]), vreinterpret_u16_u8(t3.val[0]));
    स्थिरांक uint16x4x2_t t7 = vtrn_u16(vreinterpret_u16_u8(t2.val[1]), vreinterpret_u16_u8(t3.val[1]));
    स्थिरांक uint32x2x2_t t8 = vtrn_u32(vreinterpret_u32_u16(t4.val[0]), vreinterpret_u32_u16(t6.val[0]));
    स्थिरांक uint32x2x2_t t9 = vtrn_u32(vreinterpret_u32_u16(t4.val[1]), vreinterpret_u32_u16(t6.val[1]));
    स्थिरांक uint32x2x2_t ta = vtrn_u32(vreinterpret_u32_u16(t5.val[0]), vreinterpret_u32_u16(t7.val[0]));
    स्थिरांक uint32x2x2_t tb = vtrn_u32(vreinterpret_u32_u16(t5.val[1]), vreinterpret_u32_u16(t7.val[1]));
    columns[0] = vreinterpret_u8_u32(t8.val[0]);
    columns[1] = vreinterpret_u8_u32(ta.val[0]);
    columns[2] = vreinterpret_u8_u32(t9.val[0]);
    columns[3] = vreinterpret_u8_u32(tb.val[0]);
    columns[4] = vreinterpret_u8_u32(t8.val[1]);
    columns[5] = vreinterpret_u8_u32(ta.val[1]);
    columns[6] = vreinterpret_u8_u32(t9.val[1]);
    columns[7] = vreinterpret_u8_u32(tb.val[1]);
}

स्थिर uint8x8_t repack_neon_pack_4bit_pairs(uint8x8_t first, uint8x8_t second, तार्किक वरचा) {
    स्थिरांक uint8x8x2_t separated = vuzp_u8(first, second);
    स्थिरांक uint8x8_t mask = vdup_n_u8(0x0F);
    uint8x8_t even = separated.val[0];
    uint8x8_t odd  = separated.val[1];

    जर (वरचा) {
        even = vshr_n_u8(even, 4);
        odd  = vshr_n_u8(odd, 4);
    }

    even = vand_u8(even, mask);
    odd  = vand_u8(odd, mask);
    परत vorr_u8(vshl_n_u8(odd, 4), even);
}

स्थिर शून्य repack_neon_4bit_rows(स्थिरांक uint8_t * qs[8], uint8_t * tile_dst, पूर्णांक row0) {
    uint8x8_t खालचा_rows[8];
    uint8x8_t वरचा_rows[8];
    uint8x8_t columns[8];

    साठी (पूर्णांक rr = 0; rr < 8; ++rr) {
        स्थिरांक uint8x16_t packed = vld1q_u8(qs[rr]);
        स्थिरांक uint8x8_t first = vget_खालचा_u8(packed);
        स्थिरांक uint8x8_t second = vget_वरचा_u8(packed);
        खालचा_rows[rr]  = repack_neon_pack_4bit_pairs(first, second, असत्य);
        वरचा_rows[rr] = repack_neon_pack_4bit_pairs(first, second, सत्य);
    }

    repack_neon_transpose8x8(खालचा_rows, columns);
    साठी (पूर्णांक cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + cp * 32 + row0, columns[cp]);
    }

    repack_neon_transpose8x8(वरचा_rows, columns);
    साठी (पूर्णांक cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + (cp + 8) * 32 + row0, columns[cp]);
    }
}

स्थिर शून्य repack_neon_k4_rows(स्थिरांक uint8_t * qs[8], uint8_t * tile_dst, पूर्णांक row0, तार्किक वरचा) {
    uint8x8_t first_rows[8];
    uint8x8_t second_rows[8];
    uint8x8_t columns[8];

    साठी (पूर्णांक rr = 0; rr < 8; ++rr) {
        first_rows[rr] = repack_neon_pack_4bit_pairs(vld1_u8(qs[rr]), vld1_u8(qs[rr] + 8), वरचा);
        second_rows[rr] = repack_neon_pack_4bit_pairs(vld1_u8(qs[rr] + 16), vld1_u8(qs[rr] + 24), वरचा);
    }

    repack_neon_transpose8x8(first_rows, columns);
    साठी (पूर्णांक cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + cp * 32 + row0, columns[cp]);
    }

    repack_neon_transpose8x8(second_rows, columns);
    साठी (पूर्णांक cp = 0; cp < 8; ++cp) {
        vst1_u8(tile_dst + (cp + 8) * 32 + row0, columns[cp]);
    }
}

स्थिर शून्य repack_neon_q8_rows(स्थिरांक uint8_t * qs[8], uint8_t * tile_dst, पूर्णांक row0) {
    uint8x8_t rows[8];
    uint8x8_t columns[8];

    साठी (पूर्णांक cp = 0; cp < 16; cp += 4) {
        साठी (पूर्णांक rr = 0; rr < 8; ++rr) {
            rows[rr] = vld1_u8(qs[rr] + 2 * cp);
        }
        repack_neon_transpose8x8(rows, columns);

        साठी (पूर्णांक pair = 0; pair < 4; ++pair) {
            स्थिरांक uint8x8x2_t interleaved = vzip_u8(columns[2 * pair], columns[2 * pair + 1]);
            uint8_t * dst = tile_dst + (cp + pair) * 64 + 2 * row0;
            vst1_u8(dst, interleaved.val[0]);
            vst1_u8(dst + 8, interleaved.val[1]);
        }
    }
}

स्थिर ओळीत शून्य repack_store_u32(uint8_t * dst, uint32_t v) {
    स_नकल(dst, &v, आकार(v));
}

स्थिर ओळीत uint32x4_t repack_neon_transpose_2bit_4x4(uint32x4_t w) {
    uint32x4_t t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 6)), vdupq_n_u32(0x00CC00CCu));
    w = veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 6)));
    t = vandq_u32(veorq_u32(w, vshrq_n_u32(w, 12)), vdupq_n_u32(0x0000F0F0u));
    परत veorq_u32(w, veorq_u32(t, vshlq_n_u32(t, 12)));
}

स्थिर ओळीत शून्य repack_neon_q2_0_block(स्थिरांक uint8_t * qs, uint8_t * tile, पूर्णांक row) {
    स्थिरांक uint32x4_t w = repack_neon_transpose_2bit_4x4(vreinterpretq_u32_u8(vld1q_u8(qs)));
    repack_store_u32(tile + 0 * 128 + 4 * row, vgetq_lane_u32(w, 0));
    repack_store_u32(tile + 1 * 128 + 4 * row, vgetq_lane_u32(w, 1));
    repack_store_u32(tile + 2 * 128 + 4 * row, vgetq_lane_u32(w, 2));
    repack_store_u32(tile + 3 * 128 + 4 * row, vgetq_lane_u32(w, 3));
}

स्थिर ओळीत शून्य repack_neon_q1_0_block(स्थिरांक uint8_t * qs, uint8_t * tile, पूर्णांक row) {
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

स्थिर ओळीत शून्य repack_neon_pack_2bit_plane(uint8x16_t v0, uint8x16_t v1, uint8_t * dst, पूर्णांक row) {
    स्थिरांक uint32x4x2_t z   = vuzpq_u32(vreinterpretq_u32_u8(v0), vreinterpretq_u32_u8(v1));
    स्थिरांक uint32x4_t   p01 = vorrq_u32(z.val[0], vshlq_n_u32(z.val[1], 2));
    स्थिरांक uint32x2x2_t u   = vuzp_u32(vget_खालचा_u32(p01), vget_वरचा_u32(p01));
    स्थिरांक uint32x2_t   out = vorr_u32(u.val[0], vshl_n_u32(u.val[1], 4));
    repack_store_u32(dst + 0 * 128 + row * 4, vget_lane_u32(out, 0));
    repack_store_u32(dst + 1 * 128 + row * 4, vget_lane_u32(out, 1));
}

स्थिर ओळीत शून्य repack_neon_q2_3_k_low2(स्थिरांक uint8_t * qs, पूर्णांक kt_local, uint8_t * lo_pl, पूर्णांक row) {
    स्थिरांक uint8_t *  qs_sub = qs + (kt_local / 4) * 32;
    स्थिरांक int8x16_t  shift  = vdupq_n_s8(-2 * (kt_local % 4));
    स्थिरांक uint8x16_t mask   = vdupq_n_u8(0x03);
    स्थिरांक uint8x16_t v0     = vandq_u8(vshlq_u8(vld1q_u8(qs_sub +  0), shift), mask);
    स्थिरांक uint8x16_t v1     = vandq_u8(vshlq_u8(vld1q_u8(qs_sub + 16), shift), mask);
    repack_neon_pack_2bit_plane(v0, v1, lo_pl, row);
}

स्थिर ओळीत शून्य repack_neon_q3_k_hmask(स्थिरांक uint8_t * hmask, पूर्णांक kt_local, uint8_t * neg_pl, पूर्णांक row) {
    स्थिर स्थिरांक uint32_t mask_शब्द[4] = { 0x01010101u, 0x02020202u, 0x04040404u, 0x08080808u };
    स्थिरांक uint8x16_t m0   = vreinterpretq_u8_u32(vld1q_u32(mask_शब्द));
    स्थिरांक uint8x16_t m1   = vshlq_n_u8(m0, 4);
    स्थिरांक uint8x16_t hbit = vdupq_n_u8((uint8_t) (1u << kt_local));
    स्थिरांक uint8x16_t b0   = vbicq_u8(m0, vtstq_u8(vld1q_u8(hmask +  0), hbit));
    स्थिरांक uint8x16_t b1   = vbicq_u8(m1, vtstq_u8(vld1q_u8(hmask + 16), hbit));
    स्थिरांक uint32x4_t b01  = vreinterpretq_u32_u8(vorrq_u8(b0, b1));
    स्थिरांक uint32x2_t f0   = vorr_u32(vget_खालचा_u32(b01), vget_वरचा_u32(b01));
    स्थिरांक uint32x2_t f1   = vpadd_u32(f0, f0);
    repack_store_u32(neg_pl + row * 4, vget_lane_u32(f1, 0));
}

स्थिर ओळीत शून्य repack_neon_q6_k(स्थिरांक block_q6_K * b, पूर्णांक kt_local, uint8_t * lo_pl, uint8_t * hi_pl, पूर्णांक row) {
    स्थिरांक पूर्णांक        c      = kt_local / 4;
    स्थिरांक पूर्णांक        g_sb   = kt_local % 4;
    स्थिरांक uint8_t *  ql     = b->ql + c * 64 + (g_sb & 1) * 32;
    स्थिरांक uint8_t *  qh     = b->qh + c * 32;
    स्थिरांक int8x16_t  lshift = vdupq_n_s8((g_sb >= 2) ? -4 : 0);
    स्थिरांक int8x16_t  hshift = vdupq_n_s8(-2 * g_sb);
    स्थिरांक uint8x16_t mask4  = vdupq_n_u8(0x0F);
    स्थिरांक uint8x16_t mask2  = vdupq_n_u8(0x03);

    स्थिरांक uint8x16_t lo0  = vandq_u8(vshlq_u8(vld1q_u8(ql +  0), lshift), mask4);
    स्थिरांक uint8x16_t lo1  = vandq_u8(vshlq_u8(vld1q_u8(ql + 16), lshift), mask4);
    स्थिरांक uint32x4x2_t lz = vuzpq_u32(vreinterpretq_u32_u8(lo0), vreinterpretq_u32_u8(lo1));
    स्थिरांक uint32x4_t lout = vorrq_u32(lz.val[0], vshlq_n_u32(lz.val[1], 4));
    repack_store_u32(lo_pl + 0 * 128 + row * 4, vgetq_lane_u32(lout, 0));
    repack_store_u32(lo_pl + 1 * 128 + row * 4, vgetq_lane_u32(lout, 1));
    repack_store_u32(lo_pl + 2 * 128 + row * 4, vgetq_lane_u32(lout, 2));
    repack_store_u32(lo_pl + 3 * 128 + row * 4, vgetq_lane_u32(lout, 3));

    स्थिरांक uint8x16_t hi0 = vandq_u8(vshlq_u8(vld1q_u8(qh +  0), hshift), mask2);
    स्थिरांक uint8x16_t hi1 = vandq_u8(vshlq_u8(vld1q_u8(qh + 16), hshift), mask2);
    repack_neon_pack_2bit_plane(hi0, hi1, hi_pl, row);
}

स्थिर ओळीत शून्य repack_neon_q5_k_qh(स्थिरांक uint8_t * qh, uint8_t hbit, uint8_t * plane, पूर्णांक row) {
    स्थिर स्थिरांक uint8_t mask_bytes[16] = { 1, 2, 1, 2, 1, 2, 1, 2, 4, 8, 4, 8, 4, 8, 4, 8 };
    स्थिरांक uint8x16_t m0   = vld1q_u8(mask_bytes);
    स्थिरांक uint8x16_t m1   = vshlq_n_u8(m0, 4);
    स्थिरांक uint8x16_t vbit = vdupq_n_u8(hbit);
    स्थिरांक uint8x16_t b0   = vandq_u8(vtstq_u8(vld1q_u8(qh +  0), vbit), m0);
    स्थिरांक uint8x16_t b1   = vandq_u8(vtstq_u8(vld1q_u8(qh + 16), vbit), m1);
    स्थिरांक uint8x16_t b01  = vorrq_u8(b0, b1);
    स्थिरांक uint8x8_t  fold = vorr_u8(vget_खालचा_u8(b01), vget_वरचा_u8(b01));
    स्थिरांक uint8x8_t  out  = vpadd_u8(fold, fold);
    plane[0 * 32 + row] = vget_lane_u8(out, 0);
    plane[1 * 32 + row] = vget_lane_u8(out, 1);
    plane[2 * 32 + row] = vget_lane_u8(out, 2);
    plane[3 * 32 + row] = vget_lane_u8(out, 3);
}

स्थिर ओळीत uint16x8_t repack_neon_load_f16x8(स्थिरांक uint16_t * src) {
    परत vld1q_u16(src);
}

#जर defined(__aarch64__) || defined(__ARM_FP16_FORMAT_IEEE)
स्थिर ओळीत uint16x8_t repack_neon_load_f16x8(स्थिरांक दशांश * src) {
    स्थिरांक float16x4_t lo = vcvt_f16_f32(vld1q_f32(src));
    स्थिरांक float16x4_t hi = vcvt_f16_f32(vld1q_f32(src + 4));
    परत vreinterpretq_u16_f16(vcombine_f16(lo, hi));
}

स्थिर ओळीत uint16x8_t repack_neon_load_f16x8(स्थिरांक ggml_bf16_t * src) {
    स्थिरांक uint16x8_t  bf   = vld1q_u16((स्थिरांक uint16_t *) src);
    स्थिरांक float32x4_t f_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_खालचा_u16(bf), 16));
    स्थिरांक float32x4_t f_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_वरचा_u16(bf), 16));
    स्थिरांक float16x4_t lo   = vcvt_f16_f32(f_lo);
    स्थिरांक float16x4_t hi   = vcvt_f16_f32(f_hi);
    परत vreinterpretq_u16_f16(vcombine_f16(lo, hi));
}
#अन्यथा
स्थिर ओळीत uint16x8_t repack_neon_load_f16x8(स्थिरांक दशांश * src) {
    uint16_t tmp[8];
    साठी (पूर्णांक i = 0; i < 8; ++i) {
        tmp[i] = GGML_FP32_TO_FP16(src[i]);
    }
    परत vld1q_u16(tmp);
}

स्थिर ओळीत uint16x8_t repack_neon_load_f16x8(स्थिरांक ggml_bf16_t * src) {
    uint16_t tmp[8];
    साठी (पूर्णांक i = 0; i < 8; ++i) {
        tmp[i] = GGML_FP32_TO_FP16(GGML_BF16_TO_FP32(src[i]));
    }
    परत vld1q_u16(tmp);
}
#endif

ढाँचा <typename SrcT>
स्थिर शून्य repack_neon_f16_tile(uint16_t * tile_dst, स्थिरांक SrcT * src_slice, int64_t row_stride, पूर्णांक ct, पूर्णांक kt) {
    साठी (पूर्णांक row0 = 0; row0 < 32; row0 += 4) {
        स्थिरांक SrcT * r0 = src_slice + (int64_t) (ct * 32 + row0 + 0) * row_stride + kt * 32;
        स्थिरांक SrcT * r1 = src_slice + (int64_t) (ct * 32 + row0 + 1) * row_stride + kt * 32;
        स्थिरांक SrcT * r2 = src_slice + (int64_t) (ct * 32 + row0 + 2) * row_stride + kt * 32;
        स्थिरांक SrcT * r3 = src_slice + (int64_t) (ct * 32 + row0 + 3) * row_stride + kt * 32;
        साठी (पूर्णांक cp = 0; cp < 16; cp += 4) {
            स्थिरांक uint32x4_t v0 = vreinterpretq_u32_u16(repack_neon_load_f16x8(r0 + 2 * cp));
            स्थिरांक uint32x4_t v1 = vreinterpretq_u32_u16(repack_neon_load_f16x8(r1 + 2 * cp));
            स्थिरांक uint32x4_t v2 = vreinterpretq_u32_u16(repack_neon_load_f16x8(r2 + 2 * cp));
            स्थिरांक uint32x4_t v3 = vreinterpretq_u32_u16(repack_neon_load_f16x8(r3 + 2 * cp));

            स्थिरांक uint32x2x2_t t0 = vtrn_u32(vget_खालचा_u32(v0),  vget_खालचा_u32(v1));
            स्थिरांक uint32x2x2_t t1 = vtrn_u32(vget_वरचा_u32(v0), vget_वरचा_u32(v1));
            स्थिरांक uint32x2x2_t t2 = vtrn_u32(vget_खालचा_u32(v2),  vget_खालचा_u32(v3));
            स्थिरांक uint32x2x2_t t3 = vtrn_u32(vget_वरचा_u32(v2), vget_वरचा_u32(v3));

            vst1q_u16(tile_dst + (cp + 0) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t0.val[0], t2.val[0])));
            vst1q_u16(tile_dst + (cp + 1) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t0.val[1], t2.val[1])));
            vst1q_u16(tile_dst + (cp + 2) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t1.val[0], t3.val[0])));
            vst1q_u16(tile_dst + (cp + 3) * 64 + row0 * 2, vreinterpretq_u16_u32(vcombine_u32(t1.val[1], t3.val[1])));
        }
    }
}
#endif

// --- Generic 32x32 tile grid walkers for block-32 and float types ---

ढाँचा <typename Fn>
स्थिर शून्य repack_parallel_for_ct(माप_प्रकार slice_bytes, पूर्णांक n_col_tiles, Fn && fn) {
    पूर्णांक n_threads = 1;
    जर (slice_bytes > 128 * 1024) {
        साठी (पूर्णांक nt : { 4, 3, 2 }) {
            जर (n_col_tiles % nt == 0) {
                n_threads = nt;
                थांबा;
            }
        }
    }

    जर (n_threads <= 1) {
        fn(0, n_col_tiles);
        परत;
    }

    स्थिरांक पूर्णांक chunk = n_col_tiles / n_threads;
    std::thread workers[3];
    साठी (पूर्णांक i = 0; i < n_threads - 1; ++i) {
        workers[i] = std::thread([&fn, i, chunk]() {
            fn(i * chunk, (i + 1) * chunk);
        });
    }
    fn((n_threads - 1) * chunk, n_col_tiles);
    साठी (पूर्णांक i = 0; i < n_threads - 1; ++i) {
        workers[i].join();
    }
}

ढाँचा <typename SrcElemT, पूर्णांक ELEMS_PER_KTILE, typename PackTileFn>
स्थिर शून्य repack_to_tiled_grid_32(ggml_tensor * t, स्थिरांक शून्य * data, माप_प्रकार offset, माप_प्रकार size,
                                    माप_प्रकार tile_size, PackTileFn && pack_tile) {
    स्थिरांक SrcElemT * src_matrix = (स्थिरांक SrcElemT *) data;
    स्थिरांक int64_t ne0 = t->ne[0];
    स्थिरांक int64_t ne1 = t->ne[1];
    स्थिरांक int64_t ne2 = t->ne[2];
    स्थिरांक int64_t ne3 = t->ne[3];
    स्थिरांक पूर्णांक n_col_tiles = hex_round_up(ne1, 32) / 32;
    स्थिरांक पूर्णांक n_k_tiles   = hex_round_up(ne0, 32) / 32;
    स्थिरांक माप_प्रकार matrix_size = (माप_प्रकार) n_col_tiles * n_k_tiles * tile_size;

    स्थिरांक माप_प्रकार slice_size = (माप_प्रकार) ne1 * ggml_row_size(t->type, ne0);
    स्थिरांक int64_t start_slice = offset / slice_size;
    int64_t end_slice = (offset + size + slice_size - 1) / slice_size;
    जर (end_slice > ne2 * ne3) {
        end_slice = ne2 * ne3;
    }

    स्थिरांक int64_t row_stride = (ne0 / (32 / ELEMS_PER_KTILE));

    साठी (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        स्थिरांक SrcElemT * src_slice = src_matrix + (slice_idx - start_slice) * (ne1 * row_stride);
        uint8_t * matrix_dst = (uint8_t *) t->data + slice_idx * matrix_size;

        repack_parallel_for_ct(matrix_size, n_col_tiles, [&](पूर्णांक ct0, पूर्णांक ct1) {
            साठी (पूर्णांक ct = ct0; ct < ct1; ct++) {
                साठी (पूर्णांक kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = matrix_dst + ((माप_प्रकार) ct * n_k_tiles + kt) * tile_size;
                    pack_tile(tile_dst, src_slice, ne0, ne1, row_stride, ct, kt);
                }
            }
        });
    }
}

ढाँचा <typename DstElemT, पूर्णांक ELEMS_PER_KTILE, typename UnpackTileFn>
स्थिर शून्य unpack_from_tiled_grid_32(शून्य * data, स्थिरांक ggml_tensor * t, माप_प्रकार offset, माप_प्रकार size,
                                      माप_प्रकार tile_size, UnpackTileFn && unpack_tile) {
    uint8_t * dst_bytes = (uint8_t *) data;
    स्थिरांक int64_t ne0 = t->ne[0];
    स्थिरांक int64_t ne1 = t->ne[1];
    स्थिरांक int64_t ne2 = t->ne[2];
    स्थिरांक int64_t ne3 = t->ne[3];
    स्थिरांक पूर्णांक n_col_tiles = hex_round_up(ne1, 32) / 32;
    स्थिरांक पूर्णांक n_k_tiles   = hex_round_up(ne0, 32) / 32;
    स्थिरांक माप_प्रकार matrix_size = (माप_प्रकार) n_col_tiles * n_k_tiles * tile_size;

    स्थिरांक माप_प्रकार row_size_bytes = ggml_row_size(t->type, ne0);
    स्थिरांक माप_प्रकार slice_size     = (माप_प्रकार) ne1 * row_size_bytes;
    स्थिरांक int64_t start_slice   = offset / slice_size;
    int64_t end_slice           = (offset + size + slice_size - 1) / slice_size;
    जर (end_slice > ne2 * ne3) {
        end_slice = ne2 * ne3;
    }

    स्थिरांक int64_t row_stride = (ne0 / (32 / ELEMS_PER_KTILE));

    साठी (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        स्थिरांक माप_प्रकार cur_start_byte     = (std::max)(offset, (माप_प्रकार) slice_idx * slice_size);
        स्थिरांक माप_प्रकार cur_end_byte       = (std::min)(offset + size, (माप_प्रकार) (slice_idx + 1) * slice_size);
        स्थिरांक माप_प्रकार slice_offset_start = cur_start_byte - (माप_प्रकार) slice_idx * slice_size;
        स्थिरांक माप_प्रकार slice_offset_end   = cur_end_byte - (माप_प्रकार) slice_idx * slice_size;

        स्थिरांक int64_t start_row = slice_offset_start / row_size_bytes;
        स्थिरांक int64_t end_row   = (std::min)((int64_t) ((slice_offset_end + row_size_bytes - 1) / row_size_bytes), ne1);

        स्थिरांक पूर्णांक start_ct = (पूर्णांक) (start_row / 32);
        स्थिरांक पूर्णांक end_ct   = (std::min)((पूर्णांक) ((end_row + 31) / 32), n_col_tiles);

        DstElemT * dst_slice = (DstElemT *) (dst_bytes + (cur_start_byte - offset));
        स्थिरांक uint8_t * matrix_src = (स्थिरांक uint8_t *) t->data + slice_idx * matrix_size;

        साठी (पूर्णांक ct = start_ct; ct < end_ct; ct++) {
            साठी (पूर्णांक kt = 0; kt < n_k_tiles; kt++) {
                स्थिरांक uint8_t * tile_src = matrix_src + ((माप_प्रकार) ct * n_k_tiles + kt) * tile_size;
                unpack_tile(dst_slice, tile_src, ne0, start_row, end_row, row_stride, ct, kt);
            }
        }
    }
}

// --- Block-32 Quantized Types (Q4_0, IQ4_NL, Q4_1, Q8_0, MXFP4) ---

स्थिर शून्य repack_q4_0_tiled(ggml_tensor * t, स्थिरांक शून्य * data, माप_प्रकार offset, माप_प्रकार size) {
    repack_to_tiled_grid_32<block_q4_0, 1>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_Q4_0,
        [](uint8_t * tile_dst, स्थिरांक block_q4_0 * src_slice, int64_t ne0, int64_t ne1, int64_t row_stride, पूर्णांक ct, पूर्णांक kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
#जर defined(__ARM_NEON) || defined(__aarch64__)
            जर (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                साठी (पूर्णांक row0 = 0; row0 < 32; row0 += 8) {
                    स्थिरांक uint8_t * qs[8];
                    साठी (पूर्णांक rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_slice[(ct * 32 + row0 + rr) * row_stride + kt].qs;
                    }
                    repack_neon_4bit_rows(qs, tile_dst, row0);
                }

                साठी (पूर्णांक row = 0; row < 32; ++row) {
                    scale_dst[row] = src_slice[(ct * 32 + row) * row_stride + kt].d;
                }
                परत;
            }
#endif
            uint8_t tile_quants[32][32];
            साठी (पूर्णांक row = 0; row < 32; row++) {
                स्थिरांक int64_t r = ct * 32 + row;
                जर (r < ne1 && kt < ne0 / 32) {
                    स्थिरांक block_q4_0 & b = src_slice[r * row_stride + kt];
                    unpack_nibbles_32(tile_quants[row], b.qs);
                    scale_dst[row] = b.d;
                } अन्यथा {
                    स_रखो(tile_quants[row], 8, 32);
                    scale_dst[row] = 0;
                }
            }
            transpose_nibbles_to_tile(tile_dst, tile_quants);
        });
}

स्थिर शून्य repack_tiled_q4_0(शून्य * data, स्थिरांक ggml_tensor * t, माप_प्रकार offset, माप_प्रकार size) {
    unpack_from_tiled_grid_32<block_q4_0, 1>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_Q4_0,
        [](block_q4_0 * dst_slice, स्थिरांक uint8_t * tile_src, int64_t ne0, int64_t start_row, int64_t end_row, int64_t row_stride, पूर्णांक ct, पूर्णांक kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_to_nibbles(tile_quants, tile_src);
            स्थिरांक ggml_half * scale_src = (स्थिरांक ggml_half *) (tile_src + 512);
            साठी (पूर्णांक row = 0; row < 32; row++) {
                स्थिरांक int64_t r = ct * 32 + row;
                जर (r >= start_row && r < end_row && kt < ne0 / 32) {
                    block_q4_0 & b = dst_slice[(r - start_row) * row_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[row]);
                    b.d = scale_src[row];
                }
            }
        });
}

स्थिर शून्य repack_q4_1_tiled(ggml_tensor * t, स्थिरांक शून्य * data, माप_प्रकार offset, माप_प्रकार size) {
    repack_to_tiled_grid_32<block_q4_1, 1>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_Q4_1,
        [](uint8_t * tile_dst, स्थिरांक block_q4_1 * src_slice, int64_t ne0, int64_t ne1, int64_t row_stride, पूर्णांक ct, पूर्णांक kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
#जर defined(__ARM_NEON) || defined(__aarch64__)
            जर (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                साठी (पूर्णांक row0 = 0; row0 < 32; row0 += 8) {
                    स्थिरांक uint8_t * qs[8];
                    साठी (पूर्णांक rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_slice[(ct * 32 + row0 + rr) * row_stride + kt].qs;
                    }
                    repack_neon_4bit_rows(qs, tile_dst, row0);
                }

                साठी (पूर्णांक row = 0; row < 32; ++row) {
                    स्थिरांक block_q4_1 & src_block = src_slice[(ct * 32 + row) * row_stride + kt];
                    scale_dst[2 * row + 0] = src_block.d;
                    scale_dst[2 * row + 1] = src_block.m;
                }
                परत;
            }
#endif
            uint8_t tile_quants[32][32];
            साठी (पूर्णांक row = 0; row < 32; row++) {
                स्थिरांक int64_t r = ct * 32 + row;
                जर (r < ne1 && kt < ne0 / 32) {
                    स्थिरांक block_q4_1 & b = src_slice[r * row_stride + kt];
                    unpack_nibbles_32(tile_quants[row], b.qs);
                    scale_dst[2 * row + 0] = b.d;
                    scale_dst[2 * row + 1] = b.m;
                } अन्यथा {
                    स_रखो(tile_quants[row], 0, 32);
                    scale_dst[2 * row + 0] = 0;
                    scale_dst[2 * row + 1] = 0;
                }
            }
            transpose_nibbles_to_tile(tile_dst, tile_quants);
        });
}

स्थिर शून्य repack_tiled_q4_1(शून्य * data, स्थिरांक ggml_tensor * t, माप_प्रकार offset, माप_प्रकार size) {
    unpack_from_tiled_grid_32<block_q4_1, 1>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_Q4_1,
        [](block_q4_1 * dst_slice, स्थिरांक uint8_t * tile_src, int64_t ne0, int64_t start_row, int64_t end_row, int64_t row_stride, पूर्णांक ct, पूर्णांक kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_to_nibbles(tile_quants, tile_src);
            स्थिरांक ggml_half * scale_src = (स्थिरांक ggml_half *) (tile_src + 512);
            साठी (पूर्णांक row = 0; row < 32; row++) {
                स्थिरांक int64_t r = ct * 32 + row;
                जर (r >= start_row && r < end_row && kt < ne0 / 32) {
                    block_q4_1 & b = dst_slice[(r - start_row) * row_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[row]);
                    b.d = scale_src[2 * row + 0];
                    b.m = scale_src[2 * row + 1];
                }
            }
        });
}

स्थिर शून्य repack_mxfp4_tiled(ggml_tensor * t, स्थिरांक शून्य * data, माप_प्रकार offset, माप_प्रकार size) {
    repack_to_tiled_grid_32<block_mxfp4, 1>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_MXFP4,
        [](uint8_t * tile_dst, स्थिरांक block_mxfp4 * src_slice, int64_t ne0, int64_t ne1, int64_t row_stride, पूर्णांक ct, पूर्णांक kt) {
            uint8_t * scale_dst = tile_dst + 512;
#जर defined(__ARM_NEON) || defined(__aarch64__)
            जर (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                साठी (पूर्णांक row0 = 0; row0 < 32; row0 += 8) {
                    स्थिरांक uint8_t * qs[8];
                    साठी (पूर्णांक rr = 0; rr < 8; ++rr) {
                        qs[rr] = src_slice[(ct * 32 + row0 + rr) * row_stride + kt].qs;
                    }
                    repack_neon_4bit_rows(qs, tile_dst, row0);
                }

                साठी (पूर्णांक row = 0; row < 32; ++row) {
                    scale_dst[row] = src_slice[(ct * 32 + row) * row_stride + kt].e;
                }
                परत;
            }
#endif
            uint8_t tile_quants[32][32];
            साठी (पूर्णांक row = 0; row < 32; row++) {
                स्थिरांक int64_t r = ct * 32 + row;
                जर (r < ne1 && kt < ne0 / 32) {
                    स्थिरांक block_mxfp4 & b = src_slice[r * row_stride + kt];
                    unpack_nibbles_32(tile_quants[row], b.qs);
                    scale_dst[row] = b.e;
                } अन्यथा {
                    स_रखो(tile_quants[row], 0, 32);
                    scale_dst[row] = 0;
                }
            }
            transpose_nibbles_to_tile(tile_dst, tile_quants);
        });
}

स्थिर शून्य repack_tiled_mxfp4(शून्य * data, स्थिरांक ggml_tensor * t, माप_प्रकार offset, माप_प्रकार size) {
    unpack_from_tiled_grid_32<block_mxfp4, 1>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_MXFP4,
        [](block_mxfp4 * dst_slice, स्थिरांक uint8_t * tile_src, int64_t ne0, int64_t start_row, int64_t end_row, int64_t row_stride, पूर्णांक ct, पूर्णांक kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_to_nibbles(tile_quants, tile_src);
            स्थिरांक uint8_t * scale_src = tile_src + 512;
            साठी (पूर्णांक row = 0; row < 32; row++) {
                स्थिरांक int64_t r = ct * 32 + row;
                जर (r >= start_row && r < end_row && kt < ne0 / 32) {
                    block_mxfp4 & b = dst_slice[(r - start_row) * row_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[row]);
                    b.e = scale_src[row];
                }
            }
        });
}

स्थिर शून्य repack_q8_0_tiled(ggml_tensor * t, स्थिरांक शून्य * data, माप_प्रकार offset, माप_प्रकार size) {
    repack_to_tiled_grid_32<block_q8_0, 1>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_Q8_0,
        [](uint8_t * tile_dst, स्थिरांक block_q8_0 * src_slice, int64_t ne0, int64_t ne1, int64_t row_stride, पूर्णांक ct, पूर्णांक kt) {
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 1024);
#जर defined(__ARM_NEON) || defined(__aarch64__)
            जर (ct * 32 + 32 <= ne1 && kt < ne0 / 32) {
                साठी (पूर्णांक row0 = 0; row0 < 32; row0 += 8) {
                    स्थिरांक uint8_t * qs[8];
                    साठी (पूर्णांक rr = 0; rr < 8; ++rr) {
                        qs[rr] = (स्थिरांक uint8_t *) src_slice[(ct * 32 + row0 + rr) * row_stride + kt].qs;
                    }
                    repack_neon_q8_rows(qs, tile_dst, row0);
                }

                साठी (पूर्णांक row = 0; row < 32; ++row) {
                    scale_dst[row] = src_slice[(ct * 32 + row) * row_stride + kt].d;
                }
                परत;
            }
#endif
            साठी (पूर्णांक cp = 0; cp < 16; cp++) {
                स्थिरांक पूर्णांक col0 = cp * 2;
                स्थिरांक पूर्णांक col1 = col0 + 1;
                साठी (पूर्णांक row = 0; row < 32; row++) {
                    स्थिरांक int64_t r = ct * 32 + row;
                    स्थिरांक block_q8_0 * b = (r < ne1 && kt < ne0 / 32) ? &src_slice[r * row_stride + kt] : nullptr;
                    tile_dst[cp * 64 + 2 * row + 0] = b ? b->qs[col0] : 0;
                    tile_dst[cp * 64 + 2 * row + 1] = b ? b->qs[col1] : 0;
                }
            }
            साठी (पूर्णांक row = 0; row < 32; row++) {
                स्थिरांक int64_t r = ct * 32 + row;
                scale_dst[row] = (r < ne1 && kt < ne0 / 32) ? src_slice[r * row_stride + kt].d : 0;
            }
        });
}

स्थिर शून्य repack_tiled_q8_0(शून्य * data, स्थिरांक ggml_tensor * t, माप_प्रकार offset, माप_प्रकार size) {
    unpack_from_tiled_grid_32<block_q8_0, 1>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_Q8_0,
        [](block_q8_0 * dst_slice, स्थिरांक uint8_t * tile_src, int64_t ne0, int64_t start_row, int64_t end_row, int64_t row_stride, पूर्णांक ct, पूर्णांक kt) {
            साठी (पूर्णांक cp = 0; cp < 16; cp++) {
                स्थिरांक पूर्णांक col0 = cp * 2;
                स्थिरांक पूर्णांक col1 = col0 + 1;
                साठी (पूर्णांक row = 0; row < 32; row++) {
                    स्थिरांक int64_t r = ct * 32 + row;
                    जर (r >= start_row && r < end_row && kt < ne0 / 32) {
                        block_q8_0 & b = dst_slice[(r - start_row) * row_stride + kt];
                        b.qs[col0] = tile_src[cp * 64 + 2 * row + 0];
                        b.qs[col1] = tile_src[cp * 64 + 2 * row + 1];
                    }
                }
            }
            स्थिरांक ggml_half * scale_src = (स्थिरांक ggml_half *) (tile_src + 1024);
            साठी (पूर्णांक row = 0; row < 32; row++) {
                स्थिरांक int64_t r = ct * 32 + row;
                जर (r >= start_row && r < end_row && kt < ne0 / 32) {
                    dst_slice[(r - start_row) * row_stride + kt].d = scale_src[row];
                }
            }
        });
}

// --- Float Types (F16, F32, BF16 -> f16_tiled HMX layout) ---

स्थिर शून्य repack_q2_0_tiled(ggml_tensor * t, स्थिरांक शून्य * data, माप_प्रकार offset, माप_प्रकार size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(t->ne[0] % 64 == 0);
    GGML_ASSERT(size == ggml_nbytes(t));

    स्थिरांक block_q2_0 * src = (स्थिरांक block_q2_0 *) data;
    स्थिरांक int64_t blocks_per_row = t->ne[0] / 64;
    स्थिरांक int64_t n_col_tiles = hex_round_up(t->ne[1], 32) / 32;
    स्थिरांक int64_t n_weight_tiles = blocks_per_row;
    स्थिरांक माप_प्रकार matrix_size = (माप_प्रकार) n_col_tiles * n_weight_tiles * HTP_TENSOR_TILE_SIZE_Q2_0;

    साठी (int64_t slice = 0; slice < t->ne[2] * t->ne[3]; ++slice) {
        स्थिरांक block_q2_0 * src_slice = src + slice * t->ne[1] * blocks_per_row;
        uint8_t * dst_slice = (uint8_t *) t->data + slice * matrix_size;
        स्थिरांक माप_प्रकार ct_bytes = (माप_प्रकार) n_weight_tiles * HTP_TENSOR_TILE_SIZE_Q2_0;

        repack_parallel_for_ct(matrix_size, (पूर्णांक) n_col_tiles, [&](पूर्णांक ct0, पूर्णांक ct1) {
            स_रखो(dst_slice + (माप_प्रकार) ct0 * ct_bytes, 0, (माप_प्रकार) (ct1 - ct0) * ct_bytes);
            स्थिरांक int64_t r0 = (int64_t) ct0 * 32;
            स्थिरांक int64_t r1 = (std::min)((int64_t) ct1 * 32, t->ne[1]);
            साठी (int64_t row_index = r0; row_index < r1; ++row_index) {
                स्थिरांक int64_t ct = row_index / 32;
                स्थिरांक पूर्णांक row = (पूर्णांक) (row_index % 32);
                साठी (int64_t kt = 0; kt < n_weight_tiles; ++kt) {
                    स्थिरांक block_q2_0 * block = src_slice + row_index * blocks_per_row + kt;
                    uint8_t * tile = dst_slice + (ct * n_weight_tiles + kt) * HTP_TENSOR_TILE_SIZE_Q2_0;
#जर defined(__ARM_NEON) || defined(__aarch64__)
                    repack_neon_q2_0_block(block->qs, tile, row);
#अन्यथा
                    साठी (पूर्णांक group = 0; group < 16; ++group) {
                        साठी (पूर्णांक byte = 0; byte < 4; ++byte) {
                            स्थिरांक पूर्णांक k = 4 * group + byte;
                            स्थिरांक uint8_t code = (block->qs[k / 4] >> (2 * (k % 4))) & 3;
                            tile[(group >> 2) * 128 + 4 * row + byte] |= code << (2 * (group & 3));
                        }
                    }
#endif
                    स_नकल(tile + 512 + 2 * row, &block->d, आकार(block->d));
                }
            }
        });
    }
}

स्थिर शून्य repack_tiled_q2_0(शून्य * data, स्थिरांक ggml_tensor * t, माप_प्रकार offset, माप_प्रकार size) {
    GGML_ASSERT(t->ne[0] % 64 == 0);
    GGML_ASSERT(offset + size <= ggml_nbytes(t));

    स्थिरांक माप_प्रकार first_block = offset / आकार(block_q2_0);
    स्थिरांक माप_प्रकार end_block = (offset + size + आकार(block_q2_0) - 1) / आकार(block_q2_0);
    स्थिरांक माप_प्रकार blocks_per_row = t->ne[0] / 64;
    स्थिरांक माप_प्रकार blocks_per_slice = t->ne[1] * blocks_per_row;
    स्थिरांक माप_प्रकार n_weight_tiles = blocks_per_row;
    स्थिरांक माप_प्रकार n_col_tiles = hex_round_up(t->ne[1], 32) / 32;
    स्थिरांक माप_प्रकार matrix_size = n_col_tiles * n_weight_tiles * HTP_TENSOR_TILE_SIZE_Q2_0;

    साठी (माप_प्रकार block_index = first_block; block_index < end_block; ++block_index) {
        स्थिरांक माप_प्रकार slice = block_index / blocks_per_slice;
        स्थिरांक माप_प्रकार in_slice = block_index % blocks_per_slice;
        स्थिरांक माप_प्रकार row_index = in_slice / blocks_per_row;
        स्थिरांक माप_प्रकार kt = in_slice % blocks_per_row;
        स्थिरांक माप_प्रकार ct = row_index / 32;
        स्थिरांक माप_प्रकार row = row_index % 32;
        स्थिरांक uint8_t * tile = (स्थिरांक uint8_t *) t->data + slice * matrix_size +
            (ct * n_weight_tiles + kt) * HTP_TENSOR_TILE_SIZE_Q2_0;

        block_q2_0 block = {};
        स_नकल(&block.d, tile + 512 + 2 * row, आकार(block.d));
        साठी (पूर्णांक group = 0; group < 16; ++group) {
            साठी (पूर्णांक byte = 0; byte < 4; ++byte) {
                स्थिरांक uint8_t packed = tile[(group >> 2) * 128 + 4 * row + byte];
                स्थिरांक uint8_t code = (packed >> (2 * (group & 3))) & 3;
                block.qs[group] |= code << (2 * byte);
            }
        }

        स्थिरांक माप_प्रकार block_begin = block_index * आकार(block_q2_0);
        स्थिरांक माप_प्रकार copy_begin = (std::max)(block_begin, offset);
        स्थिरांक माप_प्रकार copy_end = (std::min)(block_begin + आकार(block_q2_0), offset + size);
        स_नकल((uint8_t *) data + copy_begin - offset, (स्थिरांक uint8_t *) &block + copy_begin - block_begin, copy_end - copy_begin);
    }
}

ढाँचा <typename SrcT, typename ToFp16Fn>
स्थिर शून्य repack_float_to_f16_tiled(ggml_tensor * t, स्थिरांक शून्य * data, माप_प्रकार offset, माप_प्रकार size, ToFp16Fn && to_fp16) {
    repack_to_tiled_grid_32<SrcT, 32>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_F16,
        [&](uint8_t * tile_bytes, स्थिरांक SrcT * src_slice, int64_t ne0, int64_t ne1, int64_t row_stride, पूर्णांक ct, पूर्णांक kt) {
            uint16_t * tile_dst = (uint16_t *) tile_bytes;
#जर defined(__ARM_NEON) || defined(__aarch64__)
            जर (ct * 32 + 32 <= ne1 && kt * 32 + 32 <= ne0) {
                repack_neon_f16_tile(tile_dst, src_slice, row_stride, ct, kt);
                परत;
            }
#endif
            साठी (पूर्णांक cp = 0; cp < 16; cp++) {
                स्थिरांक int64_t c0 = kt * 32 + cp * 2;
                स्थिरांक int64_t c1 = c0 + 1;
                साठी (पूर्णांक row = 0; row < 32; row++) {
                    स्थिरांक int64_t r = ct * 32 + row;
                    स्थिरांक SrcT * r_ptr = (r < ne1) ? &src_slice[r * row_stride] : nullptr;
                    tile_dst[cp * 64 + row * 2 + 0] = (r_ptr && c0 < ne0) ? to_fp16(r_ptr[c0]) : 0;
                    tile_dst[cp * 64 + row * 2 + 1] = (r_ptr && c1 < ne0) ? to_fp16(r_ptr[c1]) : 0;
                }
            }
        });
}

ढाँचा <typename DstT, typename FromFp16Fn>
स्थिर शून्य unpack_f16_tiled_to_float(शून्य * data, स्थिरांक ggml_tensor * t, माप_प्रकार offset, माप_प्रकार size, FromFp16Fn && from_fp16) {
    unpack_from_tiled_grid_32<DstT, 32>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_F16,
        [&](DstT * dst_slice, स्थिरांक uint8_t * tile_bytes, int64_t ne0, int64_t start_row, int64_t end_row, int64_t row_stride, पूर्णांक ct, पूर्णांक kt) {
            स्थिरांक uint16_t * tile_src = (स्थिरांक uint16_t *) tile_bytes;
            साठी (पूर्णांक cp = 0; cp < 16; cp++) {
                स्थिरांक int64_t c0 = kt * 32 + cp * 2;
                स्थिरांक int64_t c1 = c0 + 1;
                साठी (पूर्णांक row = 0; row < 32; row++) {
                    स्थिरांक int64_t r = ct * 32 + row;
                    जर (r >= start_row && r < end_row) {
                        DstT * r_ptr = dst_slice + (r - start_row) * row_stride;
                        जर (c0 < ne0) r_ptr[c0] = from_fp16(tile_src[cp * 64 + row * 2 + 0]);
                        जर (c1 < ne0) r_ptr[c1] = from_fp16(tile_src[cp * 64 + row * 2 + 1]);
                    }
                }
            }
        });
}

// --- Generic K-Quant (super-block 256) grid walkers ---

ढाँचा <typename BlockT, typename PackKtFn>
स्थिर शून्य repack_kquant_to_tiled_grid(ggml_tensor * t, स्थिरांक शून्य * data, माप_प्रकार offset, माप_प्रकार size,
                                        माप_प्रकार tile_size, PackKtFn && pack_kt) {
    GGML_ASSERT(offset == 0);
    GGML_UNUSED(size);

    स्थिरांक BlockT * src_matrix = (स्थिरांक BlockT *) data;
    स्थिरांक int64_t ne0 = t->ne[0];
    स्थिरांक int64_t ne1 = t->ne[1];
    स्थिरांक int64_t ne2 = t->ne[2];
    स्थिरांक int64_t ne3 = t->ne[3];
    GGML_ASSERT(ne0 % QK_K == 0);

    स्थिरांक पूर्णांक n_col_tiles    = hex_round_up(ne1, 32) / 32;
    स्थिरांक पूर्णांक n_k_tiles      = hex_round_up(ne0, 32) / 32;
    स्थिरांक माप_प्रकार matrix_size = (माप_प्रकार) n_col_tiles * n_k_tiles * tile_size;
    स्थिरांक int64_t sb_per_row = ne0 / QK_K;

    साठी (int64_t slice_idx = 0; slice_idx < ne2 * ne3; slice_idx++) {
        स्थिरांक BlockT * src_slice = src_matrix + slice_idx * (ne1 * sb_per_row);
        uint8_t * matrix_dst     = (uint8_t *) t->data + slice_idx * matrix_size;
        स्थिरांक माप_प्रकार ct_bytes    = (माप_प्रकार) n_k_tiles * tile_size;

        repack_parallel_for_ct(matrix_size, n_col_tiles, [&](पूर्णांक ct0, पूर्णांक ct1) {
            // Padding rows and OR-ed bit planes require zeroed tiles
            स_रखो(matrix_dst + (माप_प्रकार) ct0 * ct_bytes, 0, (माप_प्रकार) (ct1 - ct0) * ct_bytes);

            स्थिरांक int64_t r0 = (int64_t) ct0 * 32;
            स्थिरांक int64_t r1 = (std::min)((int64_t) ct1 * 32, ne1);
            साठी (int64_t r = r0; r < r1; r++) {
                स्थिरांक पूर्णांक ct  = (पूर्णांक) (r / 32);
                स्थिरांक पूर्णांक row = (पूर्णांक) (r % 32);
                स्थिरांक BlockT * src_row = src_slice + r * sb_per_row;

                साठी (पूर्णांक kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile = matrix_dst + ((माप_प्रकार) ct * n_k_tiles + kt) * tile_size;
                    pack_kt(tile, &src_row[kt / 8], row, kt % 8);
                }
            }
        });
    }
}

ढाँचा <typename BlockT, typename UnpackSbFn>
स्थिर शून्य unpack_kquant_from_tiled_grid(शून्य * data, स्थिरांक ggml_tensor * t, माप_प्रकार offset, माप_प्रकार size,
                                          माप_प्रकार tile_size, UnpackSbFn && unpack_sb) {
    uint8_t * dst_bytes = (uint8_t *) data;
    स्थिरांक int64_t ne0 = t->ne[0];
    स्थिरांक int64_t ne1 = t->ne[1];
    स्थिरांक int64_t ne2 = t->ne[2];
    स्थिरांक int64_t ne3 = t->ne[3];
    GGML_ASSERT(ne0 % QK_K == 0);

    स्थिरांक पूर्णांक n_col_tiles    = hex_round_up(ne1, 32) / 32;
    स्थिरांक पूर्णांक n_k_tiles      = hex_round_up(ne0, 32) / 32;
    स्थिरांक माप_प्रकार matrix_size = (माप_प्रकार) n_col_tiles * n_k_tiles * tile_size;
    स्थिरांक int64_t sb_per_row = ne0 / QK_K;

    स्थिरांक माप_प्रकार row_size_bytes = ggml_row_size(t->type, ne0);
    स्थिरांक माप_प्रकार slice_size     = (माप_प्रकार) ne1 * row_size_bytes;
    स्थिरांक int64_t start_slice   = offset / slice_size;
    int64_t end_slice           = (offset + size + slice_size - 1) / slice_size;
    जर (end_slice > ne2 * ne3) {
        end_slice = ne2 * ne3;
    }

    साठी (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        स्थिरांक माप_प्रकार cur_start_byte     = (std::max)(offset, (माप_प्रकार) slice_idx * slice_size);
        स्थिरांक माप_प्रकार cur_end_byte       = (std::min)(offset + size, (माप_प्रकार) (slice_idx + 1) * slice_size);
        स्थिरांक माप_प्रकार slice_offset_start = cur_start_byte - (माप_प्रकार) slice_idx * slice_size;
        स्थिरांक माप_प्रकार slice_offset_end   = cur_end_byte - (माप_प्रकार) slice_idx * slice_size;

        स्थिरांक int64_t start_row = slice_offset_start / row_size_bytes;
        स्थिरांक int64_t end_row   = (std::min)((int64_t) ((slice_offset_end + row_size_bytes - 1) / row_size_bytes), ne1);

        BlockT * dst_slice         = (BlockT *) (dst_bytes + (cur_start_byte - offset));
        स्थिरांक uint8_t * matrix_src = (स्थिरांक uint8_t *) t->data + slice_idx * matrix_size;

        साठी (int64_t r = start_row; r < end_row; r++) {
            स्थिरांक पूर्णांक ct  = (पूर्णांक) (r / 32);
            स्थिरांक पूर्णांक row = (पूर्णांक) (r % 32);
            BlockT * dst_row = dst_slice + (r - start_row) * sb_per_row;

            साठी (int64_t sb = 0; sb < sb_per_row; sb++) {
                BlockT * b = &dst_row[sb];
                स_रखो(b, 0, आकार(BlockT));
                स्थिरांक uint8_t * sb_tiles = matrix_src + ((माप_प्रकार) ct * n_k_tiles + sb * 8) * tile_size;
                unpack_sb(b, sb_tiles, tile_size, row);
            }
        }
    }
}

// --- K-Quant Helpers & Per-Type Implementations (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) ---

#जर !defined(__ARM_NEON) && !defined(__aarch64__)
// unsigned 6-bit value (0..63) of element e of a Q6_K block, same bit layout as dequantize_row_q6_K
स्थिर ओळीत uint8_t q6_K_get_quant(स्थिरांक block_q6_K * b, पूर्णांक e) {
    स्थिरांक पूर्णांक c = e / 128;
    स्थिरांक पूर्णांक w = e % 128;
    स्थिरांक पूर्णांक g = w / 32;
    स्थिरांक पूर्णांक l = w % 32;
    स्थिरांक uint8_t * ql = b->ql + c * 64;
    स्थिरांक uint8_t * qh = b->qh + c * 32;
    uint8_t lo, hi;
    निवड (g) {
        स्थिती 0:  lo = ql[l]      & 0xF; hi = (qh[l] >> 0) & 3; थांबा;
        स्थिती 1:  lo = ql[l + 32] & 0xF; hi = (qh[l] >> 2) & 3; थांबा;
        स्थिती 2:  lo = ql[l]      >> 4;  hi = (qh[l] >> 4) & 3; थांबा;
        पूर्वनिर्धारित: lo = ql[l + 32] >> 4;  hi = (qh[l] >> 6) & 3; थांबा;
    }
    परत (uint8_t) (lo | (hi << 4));
}
#endif

स्थिर शून्य repack_q6_K_tiled(ggml_tensor * t, स्थिरांक शून्य * data, माप_प्रकार offset, माप_प्रकार size) {
    repack_kquant_to_tiled_grid<block_q6_K>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_Q6_K,
        [](uint8_t * tile, स्थिरांक block_q6_K * b, पूर्णांक row, पूर्णांक kt_local) {
            स्थिरांक दशांश d     = GGML_FP16_TO_FP32(b->d);
            uint8_t * lo_pl   = tile;
            uint8_t * hi_pl   = tile + 512;
            ggml_half * sc_pl = (ggml_half *) (tile + 768);

#जर defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q6_k(b, kt_local, lo_pl, hi_pl, row);
#अन्यथा
            साठी (पूर्णांक lk = 0; lk < 32; lk++) {
                स्थिरांक uint8_t q6 = q6_K_get_quant(b, kt_local * 32 + lk);
                स्थिरांक पूर्णांक g   = lk >> 2;
                स्थिरांक पूर्णांक pos = row * 4 + (lk & 3);
                lo_pl[(g >> 1) * 128 + pos] |= (uint8_t) ((q6 & 0xF) << ((g & 1) * 4));
                hi_pl[(g >> 2) * 128 + pos] |= (uint8_t) ((q6 >> 4) << ((g & 3) * 2));
            }
#endif
            साठी (पूर्णांक sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + row] = GGML_FP32_TO_FP16(d * (दशांश) b->scales[kt_local * 2 + sub]);
            }
        });
}

स्थिर शून्य repack_tiled_q6_K(शून्य * data, स्थिरांक ggml_tensor * t, माप_प्रकार offset, माप_प्रकार size) {
    unpack_kquant_from_tiled_grid<block_q6_K>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_Q6_K,
        [](block_q6_K * b, स्थिरांक uint8_t * sb_tiles, माप_प्रकार tile_size, पूर्णांक row) {
            दशांश sub_scales[16];
            साठी (पूर्णांक kt_local = 0; kt_local < 8; kt_local++) {
                स्थिरांक uint8_t *   tile  = sb_tiles + kt_local * tile_size;
                स्थिरांक uint8_t *   lo_pl = tile;
                स्थिरांक uint8_t *   hi_pl = tile + 512;
                स्थिरांक ggml_half * sc_pl = (स्थिरांक ggml_half *) (tile + 768);

                स्थिरांक पूर्णांक c  = kt_local / 4;
                स्थिरांक पूर्णांक g  = kt_local % 4;
                uint8_t * ql = b->ql + c * 64;
                uint8_t * qh = b->qh + c * 32;

                साठी (पूर्णांक lk = 0; lk < 32; lk++) {
                    स्थिरांक पूर्णांक g_tile = lk >> 2;
                    स्थिरांक पूर्णांक pos    = row * 4 + (lk & 3);
                    स्थिरांक uint8_t lo = (lo_pl[(g_tile >> 1) * 128 + pos] >> ((g_tile & 1) * 4)) & 0xF;
                    स्थिरांक uint8_t hi = (hi_pl[(g_tile >> 2) * 128 + pos] >> ((g_tile & 3) * 2)) & 3;

                    निवड (g) {
                        स्थिती 0:  ql[lk]      |= lo;        qh[lk] |= (hi << 0); थांबा;
                        स्थिती 1:  ql[lk + 32] |= lo;        qh[lk] |= (hi << 2); थांबा;
                        स्थिती 2:  ql[lk]      |= (lo << 4); qh[lk] |= (hi << 4); थांबा;
                        पूर्वनिर्धारित: ql[lk + 32] |= (lo << 4); qh[lk] |= (hi << 6); थांबा;
                    }
                }

                साठी (पूर्णांक sub = 0; sub < 2; sub++) {
                    sub_scales[kt_local * 2 + sub] = GGML_FP16_TO_FP32(sc_pl[sub * 32 + row]);
                }
            }

            दशांश max_abs_scale = 0.0f;
            साठी (पूर्णांक s = 0; s < 16; s++) {
                max_abs_scale = (std::max)(max_abs_scale, fabsf(sub_scales[s]));
            }

            जर (max_abs_scale == 0.0f) {
                b->d = GGML_FP32_TO_FP16(0.0f);
            } अन्यथा {
                b->d = GGML_FP32_TO_FP16(max_abs_scale / 127.0f);
                स्थिरांक दशांश d_actual = GGML_FP16_TO_FP32(b->d);
                स्थिरांक दशांश inv_d    = (d_actual != 0.0f) ? (1.0f / d_actual) : 0.0f;
                साठी (पूर्णांक s = 0; s < 16; s++) {
                    स्थिरांक पूर्णांक sc = (पूर्णांक) roundf(sub_scales[s] * inv_d);
                    b->scales[s] = (int8_t) (std::max)(-128, (std::min)(127, sc));
                }
            }
        });
}

#जर !defined(__ARM_NEON) && !defined(__aarch64__)
// low 2 bits (0..3) of element e of a Q2_K or Q3_K block, same bit layout as dequantize_row_q2_K / q3_K
स्थिर ओळीत uint8_t q2_3_K_get_low2(स्थिरांक uint8_t * qs, पूर्णांक e) {
    स्थिरांक पूर्णांक c = e / 128;
    स्थिरांक पूर्णांक j = (e % 128) / 32;
    स्थिरांक पूर्णांक l = e % 32;
    परत (qs[c * 32 + l] >> (2 * j)) & 3;
}

// hmask bit of element e of a Q3_K block, same bit layout as dequantize_row_q3_K
स्थिर ओळीत तार्किक q3_K_get_hbit(स्थिरांक block_q3_K * b, पूर्णांक e) {
    स्थिरांक पूर्णांक c = e / 128;
    स्थिरांक पूर्णांक j = (e % 128) / 32;
    स्थिरांक पूर्णांक l = e % 32;
    परत (b->hmask[l] >> (c * 4 + j)) & 1;
}
#endif

// signed 6-bit scale j (-32..31) of a Q3_K block, same packing as quantize_row_q3_K_ref
स्थिर ओळीत पूर्णांक q3_K_get_scale(स्थिरांक uint8_t * scales, पूर्णांक j) {
    स्थिरांक पूर्णांक lo = (j < 8) ? (scales[j] & 0xF) : (scales[j - 8] >> 4);
    स्थिरांक पूर्णांक hi = (scales[8 + j % 4] >> (2 * (j / 4))) & 3;
    परत (lo | (hi << 4)) - 32;
}

// read-back: find fp16 d and l[j] in [lmin, lmax] with fp16(d * l[j]) == prod[j] for all j, false if none
स्थिर तार्किक hexagon_recover_k_scales(स्थिरांक ggml_half * prod, पूर्णांक n, पूर्णांक lmin, पूर्णांक lmax, ggml_half * d_out, पूर्णांक * l_out) {
    पूर्णांक jmax = 0;
    साठी (पूर्णांक j = 1; j < n; j++) {
        जर (fabsf(GGML_FP16_TO_FP32(prod[j])) > fabsf(GGML_FP16_TO_FP32(prod[jmax]))) {
            jmax = j;
        }
    }
    स्थिरांक दशांश pmax = GGML_FP16_TO_FP32(prod[jmax]);
    जर (pmax == 0.0f) {
        *d_out = GGML_FP32_TO_FP16(0.0f);
        साठी (पूर्णांक j = 0; j < n; j++) {
            l_out[j] = 0;
        }
        परत सत्य;
    }

    // the quantizers put the largest scale at or near the range end, so try large |l| first
    स्थिरांक पूर्णांक lext = (std::max)(-lmin, lmax);
    साठी (पूर्णांक a = lext; a >= 1; a--) {
        साठी (पूर्णांक sign : { 1, -1 }) {
            स्थिरांक पूर्णांक lj = sign * a;
            जर (lj < lmin || lj > lmax) {
                सुरूठेवा;
            }
            स्थिरांक ggml_half d0 = GGML_FP32_TO_FP16(pmax / (दशांश) lj);
            साठी (पूर्णांक ulp : { 0, -1, 1 }) {
                ggml_half d = d0;
                uint16_t bits;
                स_नकल(&bits, &d, आकार(bits));
                bits = (uint16_t) (bits + ulp);
                स_नकल(&d, &bits, आकार(bits));

                स्थिरांक दशांश df = GGML_FP16_TO_FP32(d);
                जर (!std::है_सीमित(df) || df == 0.0f) {
                    सुरूठेवा;
                }
                तार्किक ok = सत्य;
                साठी (पूर्णांक j = 0; j < n && ok; j++) {
                    स्थिरांक पूर्णांक l = (पूर्णांक) roundf(GGML_FP16_TO_FP32(prod[j]) / df);
                    स्थिरांक ggml_half p = GGML_FP32_TO_FP16(df * (दशांश) l);
                    ok = l >= lmin && l <= lmax && स_भेद(&p, &prod[j], आकार(p)) == 0;
                    l_out[j] = l;
                }
                जर (ok) {
                    *d_out = d;
                    परत सत्य;
                }
            }
        }
    }
    परत असत्य;
}

स्थिर शून्य repack_q3_K_tiled(ggml_tensor * t, स्थिरांक शून्य * data, माप_प्रकार offset, माप_प्रकार size) {
    repack_kquant_to_tiled_grid<block_q3_K>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_Q3_K,
        [](uint8_t * tile, स्थिरांक block_q3_K * b, पूर्णांक row, पूर्णांक kt_local) {
            स्थिरांक दशांश d     = GGML_FP16_TO_FP32(b->d);
            uint8_t * lo_pl   = tile;
            uint8_t * neg_pl  = tile + 256;
            ggml_half * sc_pl = (ggml_half *) (tile + 384);

#जर defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q2_3_k_low2(b->qs, kt_local, lo_pl, row);
            repack_neon_q3_k_hmask(b->hmask, kt_local, neg_pl, row);
#अन्यथा
            साठी (पूर्णांक lk = 0; lk < 32; lk++) {
                स्थिरांक पूर्णांक e   = kt_local * 32 + lk;
                स्थिरांक पूर्णांक g   = lk >> 2;
                स्थिरांक पूर्णांक pos = row * 4 + (lk & 3);
                lo_pl[(g >> 2) * 128 + pos] |= (uint8_t) (q2_3_K_get_low2(b->qs, e) << ((g & 3) * 2));
                जर (!q3_K_get_hbit(b, e)) {
                    neg_pl[pos] |= (uint8_t) (1 << g);
                }
            }
#endif
            साठी (पूर्णांक sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + row] = GGML_FP32_TO_FP16(d * (दशांश) q3_K_get_scale(b->scales, kt_local * 2 + sub));
            }
        });
}

स्थिर शून्य repack_tiled_q3_K(शून्य * data, स्थिरांक ggml_tensor * t, माप_प्रकार offset, माप_प्रकार size) {
    unpack_kquant_from_tiled_grid<block_q3_K>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_Q3_K,
        [](block_q3_K * b, स्थिरांक uint8_t * sb_tiles, माप_प्रकार tile_size, पूर्णांक row) {
            ggml_half sub_scales[16];
            साठी (पूर्णांक kt_local = 0; kt_local < 8; kt_local++) {
                स्थिरांक uint8_t *   tile   = sb_tiles + kt_local * tile_size;
                स्थिरांक uint8_t *   lo_pl  = tile;
                स्थिरांक uint8_t *   neg_pl = tile + 256;
                स्थिरांक ggml_half * sc_pl  = (स्थिरांक ggml_half *) (tile + 384);

                साठी (पूर्णांक lk = 0; lk < 32; lk++) {
                    स्थिरांक पूर्णांक e   = kt_local * 32 + lk;
                    स्थिरांक पूर्णांक g   = lk >> 2;
                    स्थिरांक पूर्णांक pos = row * 4 + (lk & 3);
                    स्थिरांक uint8_t lo = (lo_pl[(g >> 2) * 128 + pos] >> ((g & 3) * 2)) & 3;

                    स्थिरांक पूर्णांक c = e / 128;
                    स्थिरांक पूर्णांक j = (e % 128) / 32;
                    स्थिरांक पूर्णांक l = e % 32;
                    b->qs[c * 32 + l] |= (uint8_t) (lo << (2 * j));
                    जर (!((neg_pl[pos] >> g) & 1)) {
                        b->hmask[l] |= (uint8_t) (1 << (c * 4 + j));
                    }
                }

                साठी (पूर्णांक sub = 0; sub < 2; sub++) {
                    sub_scales[kt_local * 2 + sub] = sc_pl[sub * 32 + row];
                }
            }

            पूर्णांक ls[16];
            जर (!hexagon_recover_k_scales(sub_scales, 16, -32, 31, &b->d, ls)) {
                दशांश max_scale = 0.0f;
                साठी (पूर्णांक s = 0; s < 16; s++) {
                    जर (fabsf(GGML_FP16_TO_FP32(sub_scales[s])) > fabsf(max_scale)) {
                        max_scale = GGML_FP16_TO_FP32(sub_scales[s]);
                    }
                }
                b->d = GGML_FP32_TO_FP16(-max_scale / 32.0f);
                स्थिरांक दशांश d_actual = GGML_FP16_TO_FP32(b->d);
                स्थिरांक दशांश inv_d    = (d_actual != 0.0f) ? (1.0f / d_actual) : 0.0f;
                साठी (पूर्णांक s = 0; s < 16; s++) {
                    ls[s] = (std::max)(-32, (std::min)(31, (पूर्णांक) roundf(GGML_FP16_TO_FP32(sub_scales[s]) * inv_d)));
                }
            }

            साठी (पूर्णांक s = 0; s < 16; s++) {
                स्थिरांक पूर्णांक l = ls[s] + 32;
                जर (s < 8) {
                    b->scales[s] = l & 0xF;
                } अन्यथा {
                    b->scales[s - 8] |= (uint8_t) ((l & 0xF) << 4);
                }
                b->scales[s % 4 + 8] |= (uint8_t) ((l >> 4) << (2 * (s / 4)));
            }
        });
}

स्थिर शून्य repack_q2_K_tiled(ggml_tensor * t, स्थिरांक शून्य * data, माप_प्रकार offset, माप_प्रकार size) {
    repack_kquant_to_tiled_grid<block_q2_K>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_Q2_K,
        [](uint8_t * tile, स्थिरांक block_q2_K * b, पूर्णांक row, पूर्णांक kt_local) {
            uint8_t * lo_pl   = tile;
            uint8_t * sc_pl   = tile + 256;
            ggml_half * dd_pl = (ggml_half *) (tile + 320);

#जर defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q2_3_k_low2(b->qs, kt_local, lo_pl, row);
#अन्यथा
            साठी (पूर्णांक lk = 0; lk < 32; lk++) {
                स्थिरांक पूर्णांक g   = lk >> 2;
                स्थिरांक पूर्णांक pos = row * 4 + (lk & 3);
                lo_pl[(g >> 2) * 128 + pos] |= (uint8_t) (q2_3_K_get_low2(b->qs, kt_local * 32 + lk) << ((g & 3) * 2));
            }
#endif
            साठी (पूर्णांक sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + row] = b->scales[kt_local * 2 + sub];
            }
            जर (row / 8 == kt_local % 4) {
                dd_pl[row % 8] = kt_local < 4 ? b->d : (ggml_half) (b->dmin ^ 0x8000);
            }
        });
}

स्थिर शून्य repack_tiled_q2_K(शून्य * data, स्थिरांक ggml_tensor * t, माप_प्रकार offset, माप_प्रकार size) {
    unpack_kquant_from_tiled_grid<block_q2_K>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_Q2_K,
        [](block_q2_K * b, स्थिरांक uint8_t * sb_tiles, माप_प्रकार tile_size, पूर्णांक row) {
            साठी (पूर्णांक kt_local = 0; kt_local < 8; kt_local++) {
                स्थिरांक uint8_t *   tile  = sb_tiles + kt_local * tile_size;
                स्थिरांक uint8_t *   lo_pl = tile;
                स्थिरांक uint8_t *   sc_pl = tile + 256;
                स्थिरांक ggml_half * dd_pl = (स्थिरांक ggml_half *) (tile + 320);

                साठी (पूर्णांक lk = 0; lk < 32; lk++) {
                    स्थिरांक पूर्णांक e   = kt_local * 32 + lk;
                    स्थिरांक पूर्णांक g   = lk >> 2;
                    स्थिरांक पूर्णांक pos = row * 4 + (lk & 3);
                    स्थिरांक uint8_t lo = (lo_pl[(g >> 2) * 128 + pos] >> ((g & 3) * 2)) & 3;
                    b->qs[(e / 128) * 32 + e % 32] |= (uint8_t) (lo << (2 * ((e % 128) / 32)));
                }

                साठी (पूर्णांक sub = 0; sub < 2; sub++) {
                    b->scales[kt_local * 2 + sub] = sc_pl[sub * 32 + row];
                }
                जर (row / 8 == kt_local % 4) {
                    जर (kt_local < 4) {
                        b->d = dd_pl[row % 8];
                    } अन्यथा {
                        b->dmin = (ggml_half) (dd_pl[row % 8] ^ 0x8000);
                    }
                }
            }
        });
}

स्थिर ओळीत शून्य get_scale_min_k4(पूर्णांक j, स्थिरांक uint8_t * q, uint8_t * d, uint8_t * m) {
    जर (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } अन्यथा {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

स्थिर ओळीत शून्य pack_k4_scales_mins(uint8_t * scales, ggml_half * d, ggml_half * dmin,
                                       स्थिरांक दशांश * sub_scales, स्थिरांक दशांश * sub_mins) {
    दशांश max_scale = 0.0f;
    दशांश max_min   = 0.0f;
    साठी (पूर्णांक j = 0; j < 8; j++) {
        जर (sub_scales[j] > max_scale) max_scale = sub_scales[j];
        जर (sub_mins[j]   > max_min)   max_min   = sub_mins[j];
    }

    दशांश inv_scale = 0.0f;
    जर (max_scale > 0.0f) {
        *d = GGML_FP32_TO_FP16(max_scale / 63.0f);
        स्थिरांक दशांश d_actual = GGML_FP16_TO_FP32(*d);
        inv_scale = (d_actual > 0.0f) ? (1.0f / d_actual) : 0.0f;
    } अन्यथा {
        *d = GGML_FP32_TO_FP16(0.0f);
    }

    दशांश inv_min = 0.0f;
    जर (max_min > 0.0f) {
        *dmin = GGML_FP32_TO_FP16(max_min / 63.0f);
        स्थिरांक दशांश dmin_actual = GGML_FP16_TO_FP32(*dmin);
        inv_min = (dmin_actual > 0.0f) ? (1.0f / dmin_actual) : 0.0f;
    } अन्यथा {
        *dmin = GGML_FP32_TO_FP16(0.0f);
    }

    साठी (पूर्णांक j = 0; j < 8; j++) {
        uint8_t ls = (std::min)((uint8_t) 63, (uint8_t) roundf(inv_scale * sub_scales[j]));
        uint8_t lm = (std::min)((uint8_t) 63, (uint8_t) roundf(inv_min * sub_mins[j]));
        जर (j < 4) {
            scales[j]     = ls;
            scales[j + 4] = lm;
        } अन्यथा {
            scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            scales[j - 4] |= ((ls >> 4) << 6);
            scales[j - 0] |= ((lm >> 4) << 6);
        }
    }
}

ढाँचा <typename BlockT>
स्थिर ओळीत शून्य repack_k4_quants(uint8_t * tile_dst, स्थिरांक BlockT * b, पूर्णांक row, पूर्णांक kt_local,
                                    int64_t sb_per_row, तार्किक use_neon) {
#जर defined(__ARM_NEON) || defined(__aarch64__)
    जर (use_neon) {
        जर ((row & 7) == 0) {
            स्थिरांक uint8_t * qs[8];
            साठी (पूर्णांक rr = 0; rr < 8; ++rr) {
                qs[rr] = b[rr * sb_per_row].qs + (kt_local / 2) * 32;
            }
            repack_neon_k4_rows(qs, tile_dst, row, (kt_local & 1) != 0);
        }
        परत;
    }
#अन्यथा
    GGML_UNUSED(sb_per_row);
    GGML_UNUSED(use_neon);
#endif
    स्थिरांक uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
    स्थिरांक पूर्णांक shift = (kt_local & 1) ? 4 : 0;

    साठी (पूर्णांक cp = 0; cp < 16; cp++) {
        स्थिरांक uint8_t q0 = (qs_sub[2 * cp + 0] >> shift) & 0x0F;
        स्थिरांक uint8_t q1 = (qs_sub[2 * cp + 1] >> shift) & 0x0F;
        tile_dst[cp * 32 + row] = (uint8_t) ((q1 << 4) | q0);
    }
}

स्थिर शून्य repack_q4_K_tiled(ggml_tensor * t, स्थिरांक शून्य * data, माप_प्रकार offset, माप_प्रकार size) {
    स्थिरांक int64_t sb_per_row = t->ne[0] / QK_K;
    स्थिरांक तार्किक    use_neon   = (t->ne[1] % 8 == 0);
    repack_kquant_to_tiled_grid<block_q4_K>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_Q4_K,
        [sb_per_row, use_neon](uint8_t * tile_dst, स्थिरांक block_q4_K * b, पूर्णांक row, पूर्णांक kt_local) {
            स्थिरांक दशांश d    = GGML_FP16_TO_FP32(b->d);
            स्थिरांक दशांश dmin = GGML_FP16_TO_FP32(b->dmin);

            uint8_t sc, m;
            get_scale_min_k4(kt_local, b->scales, &sc, &m);

            repack_k4_quants(tile_dst, b, row, kt_local, sb_per_row, use_neon);

            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
            scale_dst[2 * row + 0] = GGML_FP32_TO_FP16(d * (दशांश) sc);
            scale_dst[2 * row + 1] = GGML_FP32_TO_FP16(-dmin * (दशांश) m);
        });
}

स्थिर शून्य repack_tiled_q4_K(शून्य * data, स्थिरांक ggml_tensor * t, माप_प्रकार offset, माप_प्रकार size) {
    unpack_kquant_from_tiled_grid<block_q4_K>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_Q4_K,
        [](block_q4_K * b, स्थिरांक uint8_t * sb_tiles, माप_प्रकार tile_size, पूर्णांक row) {
            दशांश sub_scales[8];
            दशांश sub_mins[8];

            साठी (पूर्णांक kt_local = 0; kt_local < 8; kt_local++) {
                स्थिरांक uint8_t *   tile_src  = sb_tiles + kt_local * tile_size;
                स्थिरांक ggml_half * scale_src = (स्थिरांक ggml_half *) (tile_src + 512);

                uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
                स्थिरांक पूर्णांक shift  = (kt_local & 1) ? 4 : 0;

                साठी (पूर्णांक cp = 0; cp < 16; cp++) {
                    स्थिरांक uint8_t val = tile_src[cp * 32 + row];
                    qs_sub[2 * cp + 0] |= (uint8_t) ((val & 0x0F) << shift);
                    qs_sub[2 * cp + 1] |= (uint8_t) ((val >> 4)   << shift);
                }

                स्थिरांक दशांश D = GGML_FP16_TO_FP32(scale_src[2 * row + 0]);
                स्थिरांक दशांश M = GGML_FP16_TO_FP32(scale_src[2 * row + 1]);
                sub_scales[kt_local] = (D > 0.0f) ? D : 0.0f;
                sub_mins[kt_local]   = (-M > 0.0f) ? -M : 0.0f;
            }

            pack_k4_scales_mins(b->scales, &b->d, &b->dmin, sub_scales, sub_mins);
        });
}

स्थिर शून्य repack_q5_K_tiled(ggml_tensor * t, स्थिरांक शून्य * data, माप_प्रकार offset, माप_प्रकार size) {
    स्थिरांक int64_t sb_per_row = t->ne[0] / QK_K;
    स्थिरांक तार्किक    use_neon   = (t->ne[1] % 8 == 0);
    repack_kquant_to_tiled_grid<block_q5_K>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_Q5_K,
        [sb_per_row, use_neon](uint8_t * tile_dst, स्थिरांक block_q5_K * b, पूर्णांक row, पूर्णांक kt_local) {
            स्थिरांक दशांश d     = GGML_FP16_TO_FP32(b->d);
            स्थिरांक दशांश dmin  = GGML_FP16_TO_FP32(b->dmin);
            uint8_t * plane   = tile_dst + 640;

            uint8_t sc, m;
            get_scale_min_k4(kt_local, b->scales, &sc, &m);

            repack_k4_quants(tile_dst, b, row, kt_local, sb_per_row, use_neon);

            स्थिरांक uint8_t hbit = (uint8_t) (1 << kt_local);
#जर defined(__ARM_NEON) || defined(__aarch64__)
            repack_neon_q5_k_qh(b->qh, hbit, plane, row);
#अन्यथा
            साठी (पूर्णांक cp = 0; cp < 16; cp++) {
                स्थिरांक पूर्णांक i    = cp / 4;
                स्थिरांक पूर्णांक lane = (cp % 4) * 32 + row;
                जर (b->qh[2 * cp + 0] & hbit) {
                    plane[lane] |= (uint8_t) (1 << (2 * i));
                }
                जर (b->qh[2 * cp + 1] & hbit) {
                    plane[lane] |= (uint8_t) (1 << (2 * i + 1));
                }
            }
#endif

            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
            scale_dst[2 * row + 0] = GGML_FP32_TO_FP16(d * (दशांश) sc);
            scale_dst[2 * row + 1] = GGML_FP32_TO_FP16(-dmin * (दशांश) m);
        });
}

स्थिर शून्य repack_tiled_q5_K(शून्य * data, स्थिरांक ggml_tensor * t, माप_प्रकार offset, माप_प्रकार size) {
    unpack_kquant_from_tiled_grid<block_q5_K>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_Q5_K,
        [](block_q5_K * b, स्थिरांक uint8_t * sb_tiles, माप_प्रकार tile_size, पूर्णांक row) {
            दशांश sub_scales[8];
            दशांश sub_mins[8];

            साठी (पूर्णांक kt_local = 0; kt_local < 8; kt_local++) {
                स्थिरांक uint8_t *   tile_src  = sb_tiles + kt_local * tile_size;
                स्थिरांक uint8_t *   plane     = tile_src + 640;
                स्थिरांक ggml_half * scale_src = (स्थिरांक ggml_half *) (tile_src + 512);

                uint8_t * qs_sub   = b->qs + (kt_local / 2) * 32;
                स्थिरांक पूर्णांक shift    = (kt_local & 1) ? 4 : 0;
                स्थिरांक uint8_t hbit = (uint8_t) (1 << kt_local);

                साठी (पूर्णांक cp = 0; cp < 16; cp++) {
                    स्थिरांक uint8_t val = tile_src[cp * 32 + row];
                    qs_sub[2 * cp + 0] |= (uint8_t) ((val & 0x0F) << shift);
                    qs_sub[2 * cp + 1] |= (uint8_t) ((val >> 4)   << shift);

                    स्थिरांक पूर्णांक i    = cp / 4;
                    स्थिरांक पूर्णांक lane = (cp % 4) * 32 + row;
                    जर (plane[lane] & (1 << (2 * i))) {
                        b->qh[2 * cp + 0] |= hbit;
                    }
                    जर (plane[lane] & (1 << (2 * i + 1))) {
                        b->qh[2 * cp + 1] |= hbit;
                    }
                }

                स्थिरांक दशांश D = GGML_FP16_TO_FP32(scale_src[2 * row + 0]);
                स्थिरांक दशांश M = GGML_FP16_TO_FP32(scale_src[2 * row + 1]);
                sub_scales[kt_local] = (D > 0.0f) ? D : 0.0f;
                sub_mins[kt_local]   = (-M > 0.0f) ? -M : 0.0f;
            }

            pack_k4_scales_mins(b->scales, &b->d, &b->dmin, sub_scales, sub_mins);
        });
}

स्थिर ओळीत शून्य q1_0_tiled_bit_location(पूर्णांक k, पूर्णांक row, पूर्णांक & byte_offset, पूर्णांक & bit) {
    स्थिरांक पूर्णांक quarter = k / 32;
    स्थिरांक पूर्णांक kq      = k % 32;

    byte_offset = quarter * 128 + 4 * row + (kq % 4);
    bit         = kq / 4;
}

स्थिर शून्य repack_q1_0_tiled(ggml_tensor * t, स्थिरांक शून्य * data, माप_प्रकार offset, माप_प्रकार size) {
    स्थिरांक block_q1_0 * src_matrix = (स्थिरांक block_q1_0 *) data;
    स्थिरांक int64_t ne0 = t->ne[0];
    स्थिरांक int64_t ne1 = t->ne[1];
    स्थिरांक int64_t ne2 = t->ne[2];
    स्थिरांक int64_t ne3 = t->ne[3];
    स्थिरांक पूर्णांक n_col_tiles = hex_round_up(ne1, 32) / 32;
    स्थिरांक पूर्णांक n_k_tiles = ne0 / QK1_0;
    स्थिरांक माप_प्रकार tile_size = HTP_TENSOR_TILE_SIZE_Q1_0;
    स्थिरांक माप_प्रकार matrix_size = (माप_प्रकार) n_col_tiles * n_k_tiles * tile_size;
    स्थिरांक माप_प्रकार slice_size = ne1 * ggml_row_size(t->type, ne0);
    स्थिरांक int64_t start_slice = offset / slice_size;
    स्थिरांक int64_t end_slice = (std::min)((int64_t) ((offset + size + slice_size - 1) / slice_size), ne2 * ne3);

    GGML_ASSERT(ne0 % QK1_0 == 0);

    साठी (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        स्थिरांक block_q1_0 * src_slice = src_matrix + (slice_idx - start_slice) * ne1 * n_k_tiles;
        uint8_t * matrix_dst = (uint8_t *) t->data + slice_idx * matrix_size;

        repack_parallel_for_ct(matrix_size, n_col_tiles, [&](पूर्णांक ct0, पूर्णांक ct1) {
            साठी (पूर्णांक ct = ct0; ct < ct1; ct++) {
                साठी (पूर्णांक kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = matrix_dst + ((माप_प्रकार) ct * n_k_tiles + kt) * tile_size;
                    ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
                    स_रखो(tile_dst, 0, tile_size);

                    साठी (पूर्णांक row = 0; row < 32; row++) {
                        स्थिरांक int64_t r = ct * 32 + row;
                        जर (r >= ne1) {
                            सुरूठेवा;
                        }

                        स्थिरांक block_q1_0 & block = src_slice[r * n_k_tiles + kt];
                        scale_dst[row] = block.d;
#जर defined(__ARM_NEON) || defined(__aarch64__)
                        repack_neon_q1_0_block(block.qs, tile_dst, row);
#अन्यथा
                        साठी (पूर्णांक k = 0; k < QK1_0; k++) {
                            पूर्णांक byte_offset;
                            पूर्णांक bit;
                            q1_0_tiled_bit_location(k, row, byte_offset, bit);
                            जर (block.qs[k / 8] & (1u << (k % 8))) {
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

स्थिर शून्य repack_tiled_q1_0(शून्य * data, स्थिरांक ggml_tensor * t, माप_प्रकार offset, माप_प्रकार size) {
    block_q1_0 * dst_matrix = (block_q1_0 *) data;
    स्थिरांक int64_t ne0 = t->ne[0];
    स्थिरांक int64_t ne1 = t->ne[1];
    स्थिरांक int64_t ne2 = t->ne[2];
    स्थिरांक int64_t ne3 = t->ne[3];
    स्थिरांक पूर्णांक n_col_tiles = hex_round_up(ne1, 32) / 32;
    स्थिरांक पूर्णांक n_k_tiles = ne0 / QK1_0;
    स्थिरांक माप_प्रकार tile_size = HTP_TENSOR_TILE_SIZE_Q1_0;
    स्थिरांक माप_प्रकार matrix_size = (माप_प्रकार) n_col_tiles * n_k_tiles * tile_size;
    स्थिरांक माप_प्रकार row_size_bytes = ggml_row_size(t->type, ne0);
    स्थिरांक माप_प्रकार slice_size = ne1 * row_size_bytes;
    स्थिरांक int64_t start_slice = offset / slice_size;
    स्थिरांक int64_t end_slice = (std::min)((int64_t) ((offset + size + slice_size - 1) / slice_size), ne2 * ne3);

    GGML_ASSERT(ne0 % QK1_0 == 0);

    साठी (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        स्थिरांक माप_प्रकार cur_start_byte = (std::max)(offset, (माप_प्रकार) slice_idx * slice_size);
        स्थिरांक माप_प्रकार cur_end_byte = (std::min)(offset + size, (माप_प्रकार) (slice_idx + 1) * slice_size);
        स्थिरांक int64_t start_row = (cur_start_byte - (माप_प्रकार) slice_idx * slice_size) / row_size_bytes;
        स्थिरांक int64_t end_row = (std::min)((int64_t) ((cur_end_byte - (माप_प्रकार) slice_idx * slice_size + row_size_bytes - 1) / row_size_bytes), ne1);
        block_q1_0 * dst_slice = dst_matrix + (cur_start_byte - offset) / आकार(block_q1_0);
        स्थिरांक uint8_t * matrix_src = (स्थिरांक uint8_t *) t->data + slice_idx * matrix_size;

        साठी (पूर्णांक ct = start_row / 32; ct < (end_row + 31) / 32; ct++) {
            साठी (पूर्णांक kt = 0; kt < n_k_tiles; kt++) {
                स्थिरांक uint8_t * tile_src = matrix_src + ((माप_प्रकार) ct * n_k_tiles + kt) * tile_size;
                स्थिरांक ggml_half * scale_src = (स्थिरांक ggml_half *) (tile_src + 512);

                साठी (पूर्णांक row = 0; row < 32; row++) {
                    स्थिरांक int64_t r = ct * 32 + row;
                    जर (r < start_row || r >= end_row) {
                        सुरूठेवा;
                    }

                    block_q1_0 & block = dst_slice[(r - start_row) * n_k_tiles + kt];
                    स_रखो(block.qs, 0, आकार(block.qs));
                    block.d = scale_src[row];
                    साठी (पूर्णांक k = 0; k < QK1_0; k++) {
                        पूर्णांक byte_offset;
                        पूर्णांक bit;
                        q1_0_tiled_bit_location(k, row, byte_offset, bit);
                        जर (tile_src[byte_offset] & (1u << bit)) {
                            block.qs[k / 8] |= (uint8_t) (1u << (k % 8));
                        }
                    }
                }
            }
        }
    }
}

// --- Top-level Repack / Unpack Dispatchers ---

शून्य htp_repack_tensor_tiled(ggml_tensor * tensor, स्थिरांक शून्य * data, माप_प्रकार size) {
    निवड (tensor->type) {
        स्थिती GGML_TYPE_Q1_0:
            repack_q1_0_tiled(tensor, data, 0, size);
            थांबा;
        स्थिती GGML_TYPE_Q4_0:
        स्थिती GGML_TYPE_IQ4_NL:
            repack_q4_0_tiled(tensor, data, 0, size);
            थांबा;
        स्थिती GGML_TYPE_Q4_1:
            repack_q4_1_tiled(tensor, data, 0, size);
            थांबा;
        स्थिती GGML_TYPE_Q8_0:
            repack_q8_0_tiled(tensor, data, 0, size);
            थांबा;
        स्थिती GGML_TYPE_MXFP4:
            repack_mxfp4_tiled(tensor, data, 0, size);
            थांबा;
        स्थिती GGML_TYPE_Q2_0:
            repack_q2_0_tiled(tensor, data, 0, size);
            थांबा;
        स्थिती GGML_TYPE_Q2_K:
            repack_q2_K_tiled(tensor, data, 0, size);
            थांबा;
        स्थिती GGML_TYPE_Q3_K:
            repack_q3_K_tiled(tensor, data, 0, size);
            थांबा;
        स्थिती GGML_TYPE_Q4_K:
            repack_q4_K_tiled(tensor, data, 0, size);
            थांबा;
        स्थिती GGML_TYPE_Q5_K:
            repack_q5_K_tiled(tensor, data, 0, size);
            थांबा;
        स्थिती GGML_TYPE_Q6_K:
            repack_q6_K_tiled(tensor, data, 0, size);
            थांबा;
        स्थिती GGML_TYPE_F16:
            repack_float_to_f16_tiled<uint16_t>(tensor, data, 0, size,
                [](uint16_t v) { परत v; });
            थांबा;
        स्थिती GGML_TYPE_F32:
            repack_float_to_f16_tiled<दशांश>(tensor, data, 0, size,
                [](दशांश v) { परत (uint16_t) GGML_FP32_TO_FP16(v); });
            थांबा;
        स्थिती GGML_TYPE_BF16:
            repack_float_to_f16_tiled<ggml_bf16_t>(tensor, data, 0, size,
                [](ggml_bf16_t v) { परत (uint16_t) GGML_FP32_TO_FP16(GGML_BF16_TO_FP32(v)); });
            थांबा;
        पूर्वनिर्धारित:
            थांबा;
    }
}

शून्य htp_unpack_tensor_tiled(शून्य * data, स्थिरांक ggml_tensor * tensor, माप_प्रकार offset, माप_प्रकार size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor));

    निवड (tensor->type) {
        स्थिती GGML_TYPE_Q1_0:
            repack_tiled_q1_0(data, tensor, offset, size);
            थांबा;
        स्थिती GGML_TYPE_Q4_0:
        स्थिती GGML_TYPE_IQ4_NL:
            repack_tiled_q4_0(data, tensor, offset, size);
            थांबा;
        स्थिती GGML_TYPE_Q4_1:
            repack_tiled_q4_1(data, tensor, offset, size);
            थांबा;
        स्थिती GGML_TYPE_Q8_0:
            repack_tiled_q8_0(data, tensor, offset, size);
            थांबा;
        स्थिती GGML_TYPE_MXFP4:
            repack_tiled_mxfp4(data, tensor, offset, size);
            थांबा;
        स्थिती GGML_TYPE_Q2_0:
            repack_tiled_q2_0(data, tensor, offset, size);
            थांबा;
        स्थिती GGML_TYPE_Q2_K:
            repack_tiled_q2_K(data, tensor, offset, size);
            थांबा;
        स्थिती GGML_TYPE_Q3_K:
            repack_tiled_q3_K(data, tensor, offset, size);
            थांबा;
        स्थिती GGML_TYPE_Q4_K:
            repack_tiled_q4_K(data, tensor, offset, size);
            थांबा;
        स्थिती GGML_TYPE_Q5_K:
            repack_tiled_q5_K(data, tensor, offset, size);
            थांबा;
        स्थिती GGML_TYPE_Q6_K:
            repack_tiled_q6_K(data, tensor, offset, size);
            थांबा;
        स्थिती GGML_TYPE_F16:
            unpack_f16_tiled_to_float<uint16_t>(data, tensor, offset, size,
                [](uint16_t v) { परत v; });
            थांबा;
        स्थिती GGML_TYPE_F32:
            unpack_f16_tiled_to_float<दशांश>(data, tensor, offset, size,
                [](uint16_t v) { परत ggml_fp16_to_fp32((ggml_fp16_t) v); });
            थांबा;
        स्थिती GGML_TYPE_BF16:
            unpack_f16_tiled_to_float<ggml_bf16_t>(data, tensor, offset, size,
                [](uint16_t v) { परत ggml_fp32_to_bf16(ggml_fp16_to_fp32((ggml_fp16_t) v)); });
            थांबा;
        पूर्वनिर्धारित:
            स_नकल(data, (स्थिरांक अक्षर *) tensor->data + offset, size);
            थांबा;
    }
}
