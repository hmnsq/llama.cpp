#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_IMPL_CPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_CPP
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"

#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-cpu.h"
#ಅನ್ತರ್ಭಾವ "ggml-cpu-impl.h"
#ಅನ್ತರ್ಭಾವ "simd-mappings.h"
#ಅನ್ತರ್ಭಾವ "traits.h"

#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <cassert>

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_CPU_CLANG_WORKAROUND
#ಅನ್ತರ್ಭಾವ "../../repack.h"

#ವ್ಯಾಖ್ಯಾನಿಸು ಬಳಸದ GGML_ಬಳಸದ

ಶೂನ್ಯ ggml_quantize_mat_q8_0_4x4(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT vy, int64_t k) {
    ಪ್ರತಿಜ್ಞಾ(QK8_0 == 32);
    ಪ್ರತಿಜ್ಞಾ(k % QK8_0 == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = k / QK8_0;

    ಖಂಡ_q8_0x4 * GGML_RESTRICT y = (ಖಂಡ_q8_0x4 *) vy;

#ಆದರೆ defined(__VXE__) || defined(__VXE2__)
    float32x4_t v_src[4][8];
    ದಶಮಾಂಶ       id[4];

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; i++) {
        float32x4_t v_asrc[8];
        float32x4_t v_amax[8];

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು_iter = 0; ಅಡ್ಡಸಾಲು_iter < 4; ಅಡ್ಡಸಾಲು_iter++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; j++) v_src[ಅಡ್ಡಸಾಲು_iter][j] = vec_xl(0, x + ಅಡ್ಡಸಾಲು_iter * k + i * 32 + 4 * j);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; j++) v_asrc[j] = vec_abs(v_src[ಅಡ್ಡಸಾಲು_iter][j]);

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 4; j++) v_amax[2 * j] = vec_max(v_asrc[2 * j], v_asrc[2 * j + 1]);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 2; j++) v_amax[4 * j] = vec_max(v_amax[4 * j], v_amax[4 * j + 2]);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 1; j++) v_amax[8 * j] = vec_max(v_amax[8 * j], v_amax[8 * j + 4]);

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ amax = MAX(MAX(vec_ಹೊರತೆಗೆ(v_amax[0], 0), vec_ಹೊರತೆಗೆ(v_amax[0], 1)),
                                   MAX(vec_ಹೊರತೆಗೆ(v_amax[0], 2), vec_ಹೊರತೆಗೆ(v_amax[0], 3)));

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = amax / ((1 << 7) - 1);
            id[ಅಡ್ಡಸಾಲು_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[ಅಡ್ಡಸಾಲು_iter] = GGML_CPU_FP32_ಗಮ್ಯ_FP16(d);
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; j++) {
            /* Uses non-default rounding for vec_signed or vec_round */
            ಸ್ಥಿರಾಂಕ int32x4_t v_qs0 = vec_signed(__builtin_s390_vfisb(vec_mul(v_src[0][j], id[0]), 4, 1));
            ಸ್ಥಿರಾಂಕ int32x4_t v_qs1 = vec_signed(__builtin_s390_vfisb(vec_mul(v_src[1][j], id[1]), 4, 1));
            ಸ್ಥಿರಾಂಕ int32x4_t v_qs2 = vec_signed(__builtin_s390_vfisb(vec_mul(v_src[2][j], id[2]), 4, 1));
            ಸ್ಥಿರಾಂಕ int32x4_t v_qs3 = vec_signed(__builtin_s390_vfisb(vec_mul(v_src[3][j], id[3]), 4, 1));

            ಸ್ಥಿರಾಂಕ int16x8_t v_qs01 = vec_packs(v_qs0, v_qs1);
            ಸ್ಥಿರಾಂಕ int16x8_t v_qs23 = vec_packs(v_qs2, v_qs3);

            vec_xst(vec_packs(v_qs01, v_qs23), 0, y[i].qs + 16 * j);
        }
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ಬಳಸದ(nb);
    ಬಳಸದ(y);
    ggml_quantize_mat_q8_0_4x4_generic(x, vy, k);
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_gemv_q4_0_4x4_q8_0(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, ಪೂರ್ಣಾಂಕ nr, ಪೂರ್ಣಾಂಕ nc) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qk = QK8_0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / qk;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_interleaved = 4;

    ಪ್ರತಿಜ್ಞಾ(nr == 1);
    ಪ್ರತಿಜ್ಞಾ(n % qk == 0);
    ಪ್ರತಿಜ್ಞಾ(nc % ncols_interleaved == 0);

    ಬಳಸದ(bs);
    ಬಳಸದ(nr);

#ಆದರೆ defined(__VXE__) || defined(__VXE2__)
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * a_ptr = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *) vy;
    ದಶಮಾಂಶ * res_ptr = s;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ x = 0; x < nc / ncols_interleaved; x++) {
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0x4 * b_ptr = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0x4 *) vx + (x * nb);

        float32x4_t v_sumf = vec_splats(0.0f);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < nb; l++) {
            ಸ್ಥಿರಾಂಕ int8_t * x_qs = b_ptr[l].qs;

            ಸ್ಥಿರಾಂಕ int8x16_t v_x0 = vec_xl( 0, x_qs);
            ಸ್ಥಿರಾಂಕ int8x16_t v_x1 = vec_xl(16, x_qs);
            ಸ್ಥಿರಾಂಕ int8x16_t v_x2 = vec_xl(32, x_qs);
            ಸ್ಥಿರಾಂಕ int8x16_t v_x3 = vec_xl(48, x_qs);

            ಸ್ಥಿರಾಂಕ int8x16_t v_x0l = vec_sra(vec_sl(v_x0, 4), 4);
            ಸ್ಥಿರಾಂಕ int8x16_t v_x1l = vec_sra(vec_sl(v_x1, 4), 4);
            ಸ್ಥಿರಾಂಕ int8x16_t v_x2l = vec_sra(vec_sl(v_x2, 4), 4);
            ಸ್ಥಿರಾಂಕ int8x16_t v_x3l = vec_sra(vec_sl(v_x3, 4), 4);

            ಸ್ಥಿರಾಂಕ int8x16_t v_x0h = vec_sra(v_x0, 4);
            ಸ್ಥಿರಾಂಕ int8x16_t v_x1h = vec_sra(v_x1, 4);
            ಸ್ಥಿರಾಂಕ int8x16_t v_x2h = vec_sra(v_x2, 4);
            ಸ್ಥಿರಾಂಕ int8x16_t v_x3h = vec_sra(v_x3, 4);

            ಸ್ಥಿರಾಂಕ int8_t * y_lo = a_ptr[l].qs;
            ಸ್ಥಿರಾಂಕ int8_t * y_hi = y_lo + qk / 2;

            int16x8_t v_sumi = vec_splats((int16_t)0);

            v_sumi = ggml_vec_dot_s16(v_sumi, v_x0l, ggml_vec_xl_s8x4_splat(y_lo +  0));
            v_sumi = ggml_vec_dot_s16(v_sumi, v_x1l, ggml_vec_xl_s8x4_splat(y_lo +  4));
            v_sumi = ggml_vec_dot_s16(v_sumi, v_x2l, ggml_vec_xl_s8x4_splat(y_lo +  8));
            v_sumi = ggml_vec_dot_s16(v_sumi, v_x3l, ggml_vec_xl_s8x4_splat(y_lo + 12));

            v_sumi = ggml_vec_dot_s16(v_sumi, v_x0h, ggml_vec_xl_s8x4_splat(y_hi +  0));
            v_sumi = ggml_vec_dot_s16(v_sumi, v_x1h, ggml_vec_xl_s8x4_splat(y_hi +  4));
            v_sumi = ggml_vec_dot_s16(v_sumi, v_x2h, ggml_vec_xl_s8x4_splat(y_hi +  8));
            v_sumi = ggml_vec_dot_s16(v_sumi, v_x3h, ggml_vec_xl_s8x4_splat(y_hi + 12));

            ಸ್ಥಿರಾಂಕ float32x4_t v_yd = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(a_ptr[l].d));
            ಸ್ಥಿರಾಂಕ float32x4_t v_xd = ggml_vec_xl_f16x4_finite(b_ptr[l].d);
            ಸ್ಥಿರಾಂಕ float32x4_t v_d  = vec_mul(v_yd, v_xd);

            v_sumf = vec_madd(vec_float(vec_paddl_s16(v_sumi)), v_d, v_sumf);
        }

        vec_xst(v_sumf, 0, res_ptr + x * ncols_interleaved);
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ಬಳಸದ(nb);
    ಬಳಸದ(ncols_interleaved);
    ggml_gemv_q4_0_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_gemm_q4_0_4x4_q8_0(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, ಪೂರ್ಣಾಂಕ nr, ಪೂರ್ಣಾಂಕ nc) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qk = QK8_0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / qk;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_interleaved = 4;

    ಪ್ರತಿಜ್ಞಾ(nr % 4 == 0);
    ಪ್ರತಿಜ್ಞಾ(n % qk == 0);
    ಪ್ರತಿಜ್ಞಾ(nc % ncols_interleaved == 0);

#ಆದರೆ defined(__VXE__) || defined(__VXE2__)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ y = 0; y < nr / 4; y++) {
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0x4 * a_ptr = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0x4 *) vy + (y * nb);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ x = 0; x < nc / ncols_interleaved; x++) {
            ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0x4 * b_ptr = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0x4 *) vx + (x * nb);

            float32x4_t v_sumf[4];
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ m = 0; m < 4; m++) {
                v_sumf[m] = vec_splats(0.0f);
            }

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < nb; l++) {
                int16x8_t v_sumi0 = vec_splats((int16_t)0);
                int16x8_t v_sumi1 = vec_splats((int16_t)0);
                int16x8_t v_sumi2 = vec_splats((int16_t)0);
                int16x8_t v_sumi3 = vec_splats((int16_t)0);

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 4; k++) {
                    ಸ್ಥಿರಾಂಕ int8x16_t v_x  = vec_xl(0, b_ptr[l].qs + 16 * k);
                    ಸ್ಥಿರಾಂಕ int8x16_t v_xl = vec_sra(vec_sl(v_x, 4), 4);
                    ಸ್ಥಿರಾಂಕ int8x16_t v_xh = vec_sra(v_x, 4);

                    ಸ್ಥಿರಾಂಕ int8_t * y_lo = a_ptr[l].qs + 16 * k;
                    ಸ್ಥಿರಾಂಕ int8_t * y_hi = y_lo + qk / 2 * 4;

                    v_sumi0 = ggml_vec_dot_s16(v_sumi0, v_xl, ggml_vec_xl_s8x4_splat(y_lo +  0));
                    v_sumi1 = ggml_vec_dot_s16(v_sumi1, v_xl, ggml_vec_xl_s8x4_splat(y_lo +  4));
                    v_sumi2 = ggml_vec_dot_s16(v_sumi2, v_xl, ggml_vec_xl_s8x4_splat(y_lo +  8));
                    v_sumi3 = ggml_vec_dot_s16(v_sumi3, v_xl, ggml_vec_xl_s8x4_splat(y_lo + 12));

                    v_sumi0 = ggml_vec_dot_s16(v_sumi0, v_xh, ggml_vec_xl_s8x4_splat(y_hi +  0));
                    v_sumi1 = ggml_vec_dot_s16(v_sumi1, v_xh, ggml_vec_xl_s8x4_splat(y_hi +  4));
                    v_sumi2 = ggml_vec_dot_s16(v_sumi2, v_xh, ggml_vec_xl_s8x4_splat(y_hi +  8));
                    v_sumi3 = ggml_vec_dot_s16(v_sumi3, v_xh, ggml_vec_xl_s8x4_splat(y_hi + 12));
                }

                ಸ್ಥಿರಾಂಕ float32x4_t v_yd = __lzs_f16cx4_load(a_ptr[l].d);
                ಸ್ಥಿರಾಂಕ float32x4_t v_xd = ggml_vec_xl_f16x4_finite(b_ptr[l].d);

                v_sumf[0] = vec_madd(vec_float(vec_paddl_s16(v_sumi0)), vec_mul(v_xd, vec_splat(v_yd, 0)), v_sumf[0]);
                v_sumf[1] = vec_madd(vec_float(vec_paddl_s16(v_sumi1)), vec_mul(v_xd, vec_splat(v_yd, 1)), v_sumf[1]);
                v_sumf[2] = vec_madd(vec_float(vec_paddl_s16(v_sumi2)), vec_mul(v_xd, vec_splat(v_yd, 2)), v_sumf[2]);
                v_sumf[3] = vec_madd(vec_float(vec_paddl_s16(v_sumi3)), vec_mul(v_xd, vec_splat(v_yd, 3)), v_sumf[3]);
            }

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ m = 0; m < 4; m++) {
                vec_xst(v_sumf[m], 0, s + (y * 4 + m) * bs + x * ncols_interleaved);
            }
        }
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ಬಳಸದ(nb);
    ಬಳಸದ(ncols_interleaved);
    ggml_gemm_q4_0_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
#ಆದರೆ_ಅಂತ್ಯ
}
