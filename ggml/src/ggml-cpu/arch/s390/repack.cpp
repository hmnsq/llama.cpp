#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_IMPL_CPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_CPP
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpu.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpu-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "simd-mappings.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "traits.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਅਨ੍ਤਰ੍ਭਾਵ <cassert>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CPU_CLANG_WORKAROUND
#ਅਨ੍ਤਰ੍ਭਾਵ "../../repack.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਅਣਵਰਤਿਆ GGML_ਅਣਵਰਤਿਆ

ਖਾਲੀ ggml_quantize_mat_q8_0_4x4(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k) {
    ਪ੍ਰਤਿਜ੍ਞਾ(QK8_0 == 32);
    ਪ੍ਰਤਿਜ੍ਞਾ(k % QK8_0 == 0);
    ਅਚਲ ਪੂਰਨ nb = k / QK8_0;

    ਖੰਡ_q8_0x4 * GGML_RESTRICT y = (ਖੰਡ_q8_0x4 *) vy;

#ਜੇ defined(__VXE__) || defined(__VXE2__)
    float32x4_t v_src[4][8];
    ਦਸ਼ਮਲਵ       id[4];

    ਲਈ (ਪੂਰਨ i = 0; i < nb; i++) {
        float32x4_t v_asrc[8];
        float32x4_t v_amax[8];

        ਲਈ (ਪੂਰਨ ਪੰਕਤੀ_iter = 0; ਪੰਕਤੀ_iter < 4; ਪੰਕਤੀ_iter++) {
            ਲਈ (ਪੂਰਨ j = 0; j < 8; j++) v_src[ਪੰਕਤੀ_iter][j] = vec_xl(0, x + ਪੰਕਤੀ_iter * k + i * 32 + 4 * j);
            ਲਈ (ਪੂਰਨ j = 0; j < 8; j++) v_asrc[j] = vec_abs(v_src[ਪੰਕਤੀ_iter][j]);

            ਲਈ (ਪੂਰਨ j = 0; j < 4; j++) v_amax[2 * j] = vec_max(v_asrc[2 * j], v_asrc[2 * j + 1]);
            ਲਈ (ਪੂਰਨ j = 0; j < 2; j++) v_amax[4 * j] = vec_max(v_amax[4 * j], v_amax[4 * j + 2]);
            ਲਈ (ਪੂਰਨ j = 0; j < 1; j++) v_amax[8 * j] = vec_max(v_amax[8 * j], v_amax[8 * j + 4]);

            ਅਚਲ ਦਸ਼ਮਲਵ amax = MAX(MAX(vec_ਕੱਢੋ(v_amax[0], 0), vec_ਕੱਢੋ(v_amax[0], 1)),
                                   MAX(vec_ਕੱਢੋ(v_amax[0], 2), vec_ਕੱਢੋ(v_amax[0], 3)));

            ਅਚਲ ਦਸ਼ਮਲਵ d = amax / ((1 << 7) - 1);
            id[ਪੰਕਤੀ_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[ਪੰਕਤੀ_iter] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(d);
        }

        ਲਈ (ਪੂਰਨ j = 0; j < 8; j++) {
            /* Uses non-default rounding for vec_signed or vec_round */
            ਅਚਲ int32x4_t v_qs0 = vec_signed(__builtin_s390_vfisb(vec_mul(v_src[0][j], id[0]), 4, 1));
            ਅਚਲ int32x4_t v_qs1 = vec_signed(__builtin_s390_vfisb(vec_mul(v_src[1][j], id[1]), 4, 1));
            ਅਚਲ int32x4_t v_qs2 = vec_signed(__builtin_s390_vfisb(vec_mul(v_src[2][j], id[2]), 4, 1));
            ਅਚਲ int32x4_t v_qs3 = vec_signed(__builtin_s390_vfisb(vec_mul(v_src[3][j], id[3]), 4, 1));

            ਅਚਲ int16x8_t v_qs01 = vec_packs(v_qs0, v_qs1);
            ਅਚਲ int16x8_t v_qs23 = vec_packs(v_qs2, v_qs3);

            vec_xst(vec_packs(v_qs01, v_qs23), 0, y[i].qs + 16 * j);
        }
    }
#ਨਹੀਂਤਾਂ
    ਅਣਵਰਤਿਆ(nb);
    ਅਣਵਰਤਿਆ(y);
    ggml_quantize_mat_q8_0_4x4_generic(x, vy, k);
#ਜੇ_ਅੰਤ
}

ਖਾਲੀ ggml_gemv_q4_0_4x4_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_0;
    ਅਚਲ ਪੂਰਨ nb = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 4;

    ਪ੍ਰਤਿਜ੍ਞਾ(nr == 1);
    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(nr);

#ਜੇ defined(__VXE__) || defined(__VXE2__)
    ਅਚਲ ਖੰਡ_q8_0 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0 *) vy;
    ਦਸ਼ਮਲਵ * res_ptr = s;

    ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
        ਅਚਲ ਖੰਡ_q4_0x4 * b_ptr = (ਅਚਲ ਖੰਡ_q4_0x4 *) vx + (x * nb);

        float32x4_t v_sumf = vec_splats(0.0f);

        ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
            ਅਚਲ int8_t * x_qs = b_ptr[l].qs;

            ਅਚਲ int8x16_t v_x0 = vec_xl( 0, x_qs);
            ਅਚਲ int8x16_t v_x1 = vec_xl(16, x_qs);
            ਅਚਲ int8x16_t v_x2 = vec_xl(32, x_qs);
            ਅਚਲ int8x16_t v_x3 = vec_xl(48, x_qs);

            ਅਚਲ int8x16_t v_x0l = vec_sra(vec_sl(v_x0, 4), 4);
            ਅਚਲ int8x16_t v_x1l = vec_sra(vec_sl(v_x1, 4), 4);
            ਅਚਲ int8x16_t v_x2l = vec_sra(vec_sl(v_x2, 4), 4);
            ਅਚਲ int8x16_t v_x3l = vec_sra(vec_sl(v_x3, 4), 4);

            ਅਚਲ int8x16_t v_x0h = vec_sra(v_x0, 4);
            ਅਚਲ int8x16_t v_x1h = vec_sra(v_x1, 4);
            ਅਚਲ int8x16_t v_x2h = vec_sra(v_x2, 4);
            ਅਚਲ int8x16_t v_x3h = vec_sra(v_x3, 4);

            ਅਚਲ int8_t * y_lo = a_ptr[l].qs;
            ਅਚਲ int8_t * y_hi = y_lo + qk / 2;

            int16x8_t v_sumi = vec_splats((int16_t)0);

            v_sumi = ggml_vec_dot_s16(v_sumi, v_x0l, ggml_vec_xl_s8x4_splat(y_lo +  0));
            v_sumi = ggml_vec_dot_s16(v_sumi, v_x1l, ggml_vec_xl_s8x4_splat(y_lo +  4));
            v_sumi = ggml_vec_dot_s16(v_sumi, v_x2l, ggml_vec_xl_s8x4_splat(y_lo +  8));
            v_sumi = ggml_vec_dot_s16(v_sumi, v_x3l, ggml_vec_xl_s8x4_splat(y_lo + 12));

            v_sumi = ggml_vec_dot_s16(v_sumi, v_x0h, ggml_vec_xl_s8x4_splat(y_hi +  0));
            v_sumi = ggml_vec_dot_s16(v_sumi, v_x1h, ggml_vec_xl_s8x4_splat(y_hi +  4));
            v_sumi = ggml_vec_dot_s16(v_sumi, v_x2h, ggml_vec_xl_s8x4_splat(y_hi +  8));
            v_sumi = ggml_vec_dot_s16(v_sumi, v_x3h, ggml_vec_xl_s8x4_splat(y_hi + 12));

            ਅਚਲ float32x4_t v_yd = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_ptr[l].d));
            ਅਚਲ float32x4_t v_xd = ggml_vec_xl_f16x4_finite(b_ptr[l].d);
            ਅਚਲ float32x4_t v_d  = vec_mul(v_yd, v_xd);

            v_sumf = vec_madd(vec_float(vec_paddl_s16(v_sumi)), v_d, v_sumf);
        }

        vec_xst(v_sumf, 0, res_ptr + x * ncols_interleaved);
    }
#ਨਹੀਂਤਾਂ
    ਅਣਵਰਤਿਆ(nb);
    ਅਣਵਰਤਿਆ(ncols_interleaved);
    ggml_gemv_q4_0_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
#ਜੇ_ਅੰਤ
}

ਖਾਲੀ ggml_gemm_q4_0_4x4_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_0;
    ਅਚਲ ਪੂਰਨ nb = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 4;

    ਪ੍ਰਤਿਜ੍ਞਾ(nr % 4 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nc % ncols_interleaved == 0);

#ਜੇ defined(__VXE__) || defined(__VXE2__)
    ਲਈ (ਪੂਰਨ y = 0; y < nr / 4; y++) {
        ਅਚਲ ਖੰਡ_q8_0x4 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0x4 *) vy + (y * nb);

        ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
            ਅਚਲ ਖੰਡ_q4_0x4 * b_ptr = (ਅਚਲ ਖੰਡ_q4_0x4 *) vx + (x * nb);

            float32x4_t v_sumf[4];
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                v_sumf[m] = vec_splats(0.0f);
            }

            ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
                int16x8_t v_sumi0 = vec_splats((int16_t)0);
                int16x8_t v_sumi1 = vec_splats((int16_t)0);
                int16x8_t v_sumi2 = vec_splats((int16_t)0);
                int16x8_t v_sumi3 = vec_splats((int16_t)0);

                ਲਈ (ਪੂਰਨ k = 0; k < 4; k++) {
                    ਅਚਲ int8x16_t v_x  = vec_xl(0, b_ptr[l].qs + 16 * k);
                    ਅਚਲ int8x16_t v_xl = vec_sra(vec_sl(v_x, 4), 4);
                    ਅਚਲ int8x16_t v_xh = vec_sra(v_x, 4);

                    ਅਚਲ int8_t * y_lo = a_ptr[l].qs + 16 * k;
                    ਅਚਲ int8_t * y_hi = y_lo + qk / 2 * 4;

                    v_sumi0 = ggml_vec_dot_s16(v_sumi0, v_xl, ggml_vec_xl_s8x4_splat(y_lo +  0));
                    v_sumi1 = ggml_vec_dot_s16(v_sumi1, v_xl, ggml_vec_xl_s8x4_splat(y_lo +  4));
                    v_sumi2 = ggml_vec_dot_s16(v_sumi2, v_xl, ggml_vec_xl_s8x4_splat(y_lo +  8));
                    v_sumi3 = ggml_vec_dot_s16(v_sumi3, v_xl, ggml_vec_xl_s8x4_splat(y_lo + 12));

                    v_sumi0 = ggml_vec_dot_s16(v_sumi0, v_xh, ggml_vec_xl_s8x4_splat(y_hi +  0));
                    v_sumi1 = ggml_vec_dot_s16(v_sumi1, v_xh, ggml_vec_xl_s8x4_splat(y_hi +  4));
                    v_sumi2 = ggml_vec_dot_s16(v_sumi2, v_xh, ggml_vec_xl_s8x4_splat(y_hi +  8));
                    v_sumi3 = ggml_vec_dot_s16(v_sumi3, v_xh, ggml_vec_xl_s8x4_splat(y_hi + 12));
                }

                ਅਚਲ float32x4_t v_yd = __lzs_f16cx4_load(a_ptr[l].d);
                ਅਚਲ float32x4_t v_xd = ggml_vec_xl_f16x4_finite(b_ptr[l].d);

                v_sumf[0] = vec_madd(vec_float(vec_paddl_s16(v_sumi0)), vec_mul(v_xd, vec_splat(v_yd, 0)), v_sumf[0]);
                v_sumf[1] = vec_madd(vec_float(vec_paddl_s16(v_sumi1)), vec_mul(v_xd, vec_splat(v_yd, 1)), v_sumf[1]);
                v_sumf[2] = vec_madd(vec_float(vec_paddl_s16(v_sumi2)), vec_mul(v_xd, vec_splat(v_yd, 2)), v_sumf[2]);
                v_sumf[3] = vec_madd(vec_float(vec_paddl_s16(v_sumi3)), vec_mul(v_xd, vec_splat(v_yd, 3)), v_sumf[3]);
            }

            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                vec_xst(v_sumf[m], 0, s + (y * 4 + m) * bs + x * ncols_interleaved);
            }
        }
    }
#ਨਹੀਂਤਾਂ
    ਅਣਵਰਤਿਆ(nb);
    ਅਣਵਰਤਿਆ(ncols_interleaved);
    ggml_gemm_q4_0_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
#ਜੇ_ਅੰਤ
}
