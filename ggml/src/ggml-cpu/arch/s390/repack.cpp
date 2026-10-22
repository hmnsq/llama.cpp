#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#અન્તર્ભાવ "ggml-common.h"
#અન્તર્ભાવ "ggml-backend-impl.h"

#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "ggml-cpu.h"
#અન્તર્ભાવ "ggml-cpu-impl.h"
#અન્તર્ભાવ "simd-mappings.h"
#અન્તર્ભાવ "traits.h"

#અન્તર્ભાવ <cmath>
#અન્તર્ભાવ <cstring>
#અન્તર્ભાવ <cassert>

#define GGML_CPU_CLANG_WORKAROUND
#અન્તર્ભાવ "../../repack.h"

#define UNUSED GGML_UNUSED

શૂન્ય ggml_quantize_mat_q8_0_4x4(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT vy, int64_t k) {
    પ્રતિજ્ઞા(QK8_0 == 32);
    પ્રતિજ્ઞા(k % QK8_0 == 0);
    અચલ પૂર્ણાંક nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

#જો defined(__VXE__) || defined(__VXE2__)
    float32x4_t v_src[4][8];
    દશાંશ       id[4];

    માટે (પૂર્ણાંક i = 0; i < nb; i++) {
        float32x4_t v_asrc[8];
        float32x4_t v_amax[8];

        માટે (પૂર્ણાંક row_iter = 0; row_iter < 4; row_iter++) {
            માટે (પૂર્ણાંક j = 0; j < 8; j++) v_src[row_iter][j] = vec_xl(0, x + row_iter * k + i * 32 + 4 * j);
            માટે (પૂર્ણાંક j = 0; j < 8; j++) v_asrc[j] = vec_abs(v_src[row_iter][j]);

            માટે (પૂર્ણાંક j = 0; j < 4; j++) v_amax[2 * j] = vec_max(v_asrc[2 * j], v_asrc[2 * j + 1]);
            માટે (પૂર્ણાંક j = 0; j < 2; j++) v_amax[4 * j] = vec_max(v_amax[4 * j], v_amax[4 * j + 2]);
            માટે (પૂર્ણાંક j = 0; j < 1; j++) v_amax[8 * j] = vec_max(v_amax[8 * j], v_amax[8 * j + 4]);

            અચલ દશાંશ amax = MAX(MAX(vec_extract(v_amax[0], 0), vec_extract(v_amax[0], 1)),
                                   MAX(vec_extract(v_amax[0], 2), vec_extract(v_amax[0], 3)));

            અચલ દશાંશ d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        માટે (પૂર્ણાંક j = 0; j < 8; j++) {
            /* Uses non-default rounding for vec_signed or vec_round */
            અચલ int32x4_t v_qs0 = vec_signed(__builtin_s390_vfisb(vec_mul(v_src[0][j], id[0]), 4, 1));
            અચલ int32x4_t v_qs1 = vec_signed(__builtin_s390_vfisb(vec_mul(v_src[1][j], id[1]), 4, 1));
            અચલ int32x4_t v_qs2 = vec_signed(__builtin_s390_vfisb(vec_mul(v_src[2][j], id[2]), 4, 1));
            અચલ int32x4_t v_qs3 = vec_signed(__builtin_s390_vfisb(vec_mul(v_src[3][j], id[3]), 4, 1));

            અચલ int16x8_t v_qs01 = vec_packs(v_qs0, v_qs1);
            અચલ int16x8_t v_qs23 = vec_packs(v_qs2, v_qs3);

            vec_xst(vec_packs(v_qs01, v_qs23), 0, y[i].qs + 16 * j);
        }
    }
#નહીંતર
    UNUSED(nb);
    UNUSED(y);
    ggml_quantize_mat_q8_0_4x4_generic(x, vy, k);
#endif
}

શૂન્ય ggml_gemv_q4_0_4x4_q8_0(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, અચલ શૂન્ય * GGML_RESTRICT vy, પૂર્ણાંક nr, પૂર્ણાંક nc) {
    અચલ પૂર્ણાંક qk = QK8_0;
    અચલ પૂર્ણાંક nb = n / qk;
    અચલ પૂર્ણાંક ncols_interleaved = 4;

    પ્રતિજ્ઞા(nr == 1);
    પ્રતિજ્ઞા(n % qk == 0);
    પ્રતિજ્ઞા(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

#જો defined(__VXE__) || defined(__VXE2__)
    અચલ block_q8_0 * a_ptr = (અચલ block_q8_0 *) vy;
    દશાંશ * res_ptr = s;

    માટે (પૂર્ણાંક x = 0; x < nc / ncols_interleaved; x++) {
        અચલ block_q4_0x4 * b_ptr = (અચલ block_q4_0x4 *) vx + (x * nb);

        float32x4_t v_sumf = vec_splats(0.0f);

        માટે (પૂર્ણાંક l = 0; l < nb; l++) {
            અચલ int8_t * x_qs = b_ptr[l].qs;

            અચલ int8x16_t v_x0 = vec_xl( 0, x_qs);
            અચલ int8x16_t v_x1 = vec_xl(16, x_qs);
            અચલ int8x16_t v_x2 = vec_xl(32, x_qs);
            અચલ int8x16_t v_x3 = vec_xl(48, x_qs);

            અચલ int8x16_t v_x0l = vec_sra(vec_sl(v_x0, 4), 4);
            અચલ int8x16_t v_x1l = vec_sra(vec_sl(v_x1, 4), 4);
            અચલ int8x16_t v_x2l = vec_sra(vec_sl(v_x2, 4), 4);
            અચલ int8x16_t v_x3l = vec_sra(vec_sl(v_x3, 4), 4);

            અચલ int8x16_t v_x0h = vec_sra(v_x0, 4);
            અચલ int8x16_t v_x1h = vec_sra(v_x1, 4);
            અચલ int8x16_t v_x2h = vec_sra(v_x2, 4);
            અચલ int8x16_t v_x3h = vec_sra(v_x3, 4);

            અચલ int8_t * y_lo = a_ptr[l].qs;
            અચલ int8_t * y_hi = y_lo + qk / 2;

            int16x8_t v_sumi = vec_splats((int16_t)0);

            v_sumi = ggml_vec_dot_s16(v_sumi, v_x0l, ggml_vec_xl_s8x4_splat(y_lo +  0));
            v_sumi = ggml_vec_dot_s16(v_sumi, v_x1l, ggml_vec_xl_s8x4_splat(y_lo +  4));
            v_sumi = ggml_vec_dot_s16(v_sumi, v_x2l, ggml_vec_xl_s8x4_splat(y_lo +  8));
            v_sumi = ggml_vec_dot_s16(v_sumi, v_x3l, ggml_vec_xl_s8x4_splat(y_lo + 12));

            v_sumi = ggml_vec_dot_s16(v_sumi, v_x0h, ggml_vec_xl_s8x4_splat(y_hi +  0));
            v_sumi = ggml_vec_dot_s16(v_sumi, v_x1h, ggml_vec_xl_s8x4_splat(y_hi +  4));
            v_sumi = ggml_vec_dot_s16(v_sumi, v_x2h, ggml_vec_xl_s8x4_splat(y_hi +  8));
            v_sumi = ggml_vec_dot_s16(v_sumi, v_x3h, ggml_vec_xl_s8x4_splat(y_hi + 12));

            અચલ float32x4_t v_yd = vec_splats(GGML_CPU_FP16_TO_FP32(a_ptr[l].d));
            અચલ float32x4_t v_xd = ggml_vec_xl_f16x4_finite(b_ptr[l].d);
            અચલ float32x4_t v_d  = vec_mul(v_yd, v_xd);

            v_sumf = vec_madd(vec_float(vec_paddl_s16(v_sumi)), v_d, v_sumf);
        }

        vec_xst(v_sumf, 0, res_ptr + x * ncols_interleaved);
    }
#નહીંતર
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    ggml_gemv_q4_0_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

શૂન્ય ggml_gemm_q4_0_4x4_q8_0(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, અચલ શૂન્ય * GGML_RESTRICT vy, પૂર્ણાંક nr, પૂર્ણાંક nc) {
    અચલ પૂર્ણાંક qk = QK8_0;
    અચલ પૂર્ણાંક nb = n / qk;
    અચલ પૂર્ણાંક ncols_interleaved = 4;

    પ્રતિજ્ઞા(nr % 4 == 0);
    પ્રતિજ્ઞા(n % qk == 0);
    પ્રતિજ્ઞા(nc % ncols_interleaved == 0);

#જો defined(__VXE__) || defined(__VXE2__)
    માટે (પૂર્ણાંક y = 0; y < nr / 4; y++) {
        અચલ block_q8_0x4 * a_ptr = (અચલ block_q8_0x4 *) vy + (y * nb);

        માટે (પૂર્ણાંક x = 0; x < nc / ncols_interleaved; x++) {
            અચલ block_q4_0x4 * b_ptr = (અચલ block_q4_0x4 *) vx + (x * nb);

            float32x4_t v_sumf[4];
            માટે (પૂર્ણાંક m = 0; m < 4; m++) {
                v_sumf[m] = vec_splats(0.0f);
            }

            માટે (પૂર્ણાંક l = 0; l < nb; l++) {
                int16x8_t v_sumi0 = vec_splats((int16_t)0);
                int16x8_t v_sumi1 = vec_splats((int16_t)0);
                int16x8_t v_sumi2 = vec_splats((int16_t)0);
                int16x8_t v_sumi3 = vec_splats((int16_t)0);

                માટે (પૂર્ણાંક k = 0; k < 4; k++) {
                    અચલ int8x16_t v_x  = vec_xl(0, b_ptr[l].qs + 16 * k);
                    અચલ int8x16_t v_xl = vec_sra(vec_sl(v_x, 4), 4);
                    અચલ int8x16_t v_xh = vec_sra(v_x, 4);

                    અચલ int8_t * y_lo = a_ptr[l].qs + 16 * k;
                    અચલ int8_t * y_hi = y_lo + qk / 2 * 4;

                    v_sumi0 = ggml_vec_dot_s16(v_sumi0, v_xl, ggml_vec_xl_s8x4_splat(y_lo +  0));
                    v_sumi1 = ggml_vec_dot_s16(v_sumi1, v_xl, ggml_vec_xl_s8x4_splat(y_lo +  4));
                    v_sumi2 = ggml_vec_dot_s16(v_sumi2, v_xl, ggml_vec_xl_s8x4_splat(y_lo +  8));
                    v_sumi3 = ggml_vec_dot_s16(v_sumi3, v_xl, ggml_vec_xl_s8x4_splat(y_lo + 12));

                    v_sumi0 = ggml_vec_dot_s16(v_sumi0, v_xh, ggml_vec_xl_s8x4_splat(y_hi +  0));
                    v_sumi1 = ggml_vec_dot_s16(v_sumi1, v_xh, ggml_vec_xl_s8x4_splat(y_hi +  4));
                    v_sumi2 = ggml_vec_dot_s16(v_sumi2, v_xh, ggml_vec_xl_s8x4_splat(y_hi +  8));
                    v_sumi3 = ggml_vec_dot_s16(v_sumi3, v_xh, ggml_vec_xl_s8x4_splat(y_hi + 12));
                }

                અચલ float32x4_t v_yd = __lzs_f16cx4_load(a_ptr[l].d);
                અચલ float32x4_t v_xd = ggml_vec_xl_f16x4_finite(b_ptr[l].d);

                v_sumf[0] = vec_madd(vec_float(vec_paddl_s16(v_sumi0)), vec_mul(v_xd, vec_splat(v_yd, 0)), v_sumf[0]);
                v_sumf[1] = vec_madd(vec_float(vec_paddl_s16(v_sumi1)), vec_mul(v_xd, vec_splat(v_yd, 1)), v_sumf[1]);
                v_sumf[2] = vec_madd(vec_float(vec_paddl_s16(v_sumi2)), vec_mul(v_xd, vec_splat(v_yd, 2)), v_sumf[2]);
                v_sumf[3] = vec_madd(vec_float(vec_paddl_s16(v_sumi3)), vec_mul(v_xd, vec_splat(v_yd, 3)), v_sumf[3]);
            }

            માટે (પૂર્ણાંક m = 0; m < 4; m++) {
                vec_xst(v_sumf[m], 0, s + (y * 4 + m) * bs + x * ncols_interleaved);
            }
        }
    }
#નહીંતર
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    ggml_gemm_q4_0_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
#endif
}
