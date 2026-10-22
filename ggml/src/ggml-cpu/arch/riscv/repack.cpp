#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#অন্তর্ভাব "ggml-common.h"
#অন্তর্ভাব "ggml-backend-impl.h"

#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "ggml-cpu.h"
#অন্তর্ভাব "ggml-cpu-impl.h"
#অন্তর্ভাব "simd-mappings.h"
#অন্তর্ভাব "traits.h"

#অন্তর্ভাব <cmath>
#অন্তর্ভাব <cstring>
#অন্তর্ভাব <cassert>
#অন্তর্ভাব <cstdlib> // for qsort
#অন্তর্ভাব <cstdio>  // for GGML_ASSERT

#define GGML_CPU_CLANG_WORKAROUND
#অন্তর্ভাব "../../repack.h"

#যদি defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif

#define UNUSED GGML_UNUSED

শূন্য ggml_quantize_mat_q8_0_4x8(ধ্রুবক দশমিক * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t k) {
    প্রতিজ্ঞা(QK8_0 == 32);
    প্রতিজ্ঞা(k % QK8_0 == 0);
    ধ্রুবক পূর্ণসংখ্যা nb = k / QK8_0;

#যদি defined(__riscv_v_intrinsic)
    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;
    ধ্রুবক size_t vl_calc = __riscv_vsetvl_e32m8(QK8_0);
    ধ্রুবক size_t vl_save = __riscv_vsetvl_e64m2(4);
    vfloat32m1_t v_scalar_zero = __riscv_vfmv_s_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));

    জন্য (পূর্ণসংখ্যা i = 0; i < nb; i++) {
        ধ্রুবক দশমিক *x_block_base = x + i * QK8_0;
        vint8m2_t q_r0, q_r1, q_r2, q_r3;
        {
            vfloat32m8_t v_src = __riscv_vle32_v_f32m8(x_block_base + 0 * k, vl_calc);
            vfloat32m8_t v_abs = __riscv_vfabs_v_f32m8(v_src, vl_calc);
            vfloat32m1_t v_max = __riscv_vfredmax_vs_f32m8_f32m1(v_abs, v_scalar_zero, vl_calc);
            দশমিক amax = __riscv_vfmv_f_s_f32m1_f32(v_max);

            দশমিক d = amax / 127.0f;
            y[i].d[0] = GGML_CPU_FP32_TO_FP16(d);

            দশমিক id = d ? 1.0f / d : 0.0f;
            vfloat32m8_t v_scaled = __riscv_vfmul_vf_f32m8(v_src, id, vl_calc);
            vint16m4_t v_i16 = __riscv_vfncvt_x_f_w_i16m4_rm(v_scaled, 4, vl_calc);
            q_r0 = __riscv_vncvt_x_x_w_i8m2(v_i16, vl_calc);
        }
        asm অস্থির ("" ::: "memory");

        {
            vfloat32m8_t v_src = __riscv_vle32_v_f32m8(x_block_base + 1 * k, vl_calc);
            vfloat32m8_t v_abs = __riscv_vfabs_v_f32m8(v_src, vl_calc);
            vfloat32m1_t v_max = __riscv_vfredmax_vs_f32m8_f32m1(v_abs, v_scalar_zero, vl_calc);
            দশমিক amax = __riscv_vfmv_f_s_f32m1_f32(v_max);

            দশমিক d = amax / 127.0f;
            y[i].d[1] = GGML_CPU_FP32_TO_FP16(d);
            দশমিক id = d ? 1.0f / d : 0.0f;

            vfloat32m8_t v_scaled = __riscv_vfmul_vf_f32m8(v_src, id, vl_calc);
            vint16m4_t v_i16 = __riscv_vfncvt_x_f_w_i16m4_rm(v_scaled, 4, vl_calc);
            q_r1 = __riscv_vncvt_x_x_w_i8m2(v_i16, vl_calc);
        }
        asm অস্থির ("" ::: "memory");
        {
            vfloat32m8_t v_src = __riscv_vle32_v_f32m8(x_block_base + 2 * k, vl_calc);
            vfloat32m8_t v_abs = __riscv_vfabs_v_f32m8(v_src, vl_calc);
            vfloat32m1_t v_max = __riscv_vfredmax_vs_f32m8_f32m1(v_abs, v_scalar_zero, vl_calc);
            দশমিক amax = __riscv_vfmv_f_s_f32m1_f32(v_max);

            দশমিক d = amax / 127.0f;
            y[i].d[2] = GGML_CPU_FP32_TO_FP16(d);
            দশমিক id = d ? 1.0f / d : 0.0f;

            vfloat32m8_t v_scaled = __riscv_vfmul_vf_f32m8(v_src, id, vl_calc);
            vint16m4_t v_i16 = __riscv_vfncvt_x_f_w_i16m4_rm(v_scaled, 4, vl_calc);
            q_r2 = __riscv_vncvt_x_x_w_i8m2(v_i16, vl_calc);
        }
        asm অস্থির ("" ::: "memory");
        {
            vfloat32m8_t v_src = __riscv_vle32_v_f32m8(x_block_base + 3 * k, vl_calc);
            vfloat32m8_t v_abs = __riscv_vfabs_v_f32m8(v_src, vl_calc);
            vfloat32m1_t v_max = __riscv_vfredmax_vs_f32m8_f32m1(v_abs, v_scalar_zero, vl_calc);
            দশমিক amax = __riscv_vfmv_f_s_f32m1_f32(v_max);

            দশমিক d = amax / 127.0f;
            y[i].d[3] = GGML_CPU_FP32_TO_FP16(d);
            দশমিক id = d ? 1.0f / d : 0.0f;

            vfloat32m8_t v_scaled = __riscv_vfmul_vf_f32m8(v_src, id, vl_calc);
            vint16m4_t v_i16 = __riscv_vfncvt_x_f_w_i16m4_rm(v_scaled, 4, vl_calc);
            q_r3 = __riscv_vncvt_x_x_w_i8m2(v_i16, vl_calc);
        }
        vint64m2_t v_q64_r0 = __riscv_vreinterpret_v_i8m2_i64m2(q_r0);
        vint64m2_t v_q64_r1 = __riscv_vreinterpret_v_i8m2_i64m2(q_r1);
        vint64m2_t v_q64_r2 = __riscv_vreinterpret_v_i8m2_i64m2(q_r2);
        vint64m2_t v_q64_r3 = __riscv_vreinterpret_v_i8m2_i64m2(q_r3);
        vint64m2x4_t v_quant_tuple = __riscv_vcreate_v_i64m2x4(v_q64_r0, v_q64_r1, v_q64_r2, v_q64_r3);
        __riscv_vsseg4e64_v_i64m2x4((int64_t*)y[i].qs, v_quant_tuple, vl_save);
    }
#অন্যথা
    UNUSED(nb);
    ggml_quantize_mat_q8_0_4x8_generic(x, vy, k);
#endif
}

শূন্য ggml_gemv_q4_0_8x8_q8_0(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 8;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 8;

    প্রতিজ্ঞা (n % qk == 0);
    প্রতিজ্ঞা (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#যদি defined __riscv_v
    যদি (__riscv_vlenb() >= QK4_0) {
        ধ্রুবক size_t vl = QK4_0;

        ধ্রুবক block_q8_0 * a_ptr = (ধ্রুবক block_q8_0 *) vy;
        জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
            ধ্রুবক block_q4_0x8 * b_ptr = (ধ্রুবক block_q4_0x8 *) vx + (x * nb);

            vfloat32m1_t sumf = __riscv_vfmv_v_f_f32m1(0.0, vl / 4);
            জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
                ধ্রুবক int64_t a0 = *(ধ্রুবক int64_t *)&a_ptr[l].qs[0];
                ধ্রুবক int64_t a1 = *(ধ্রুবক int64_t *)&a_ptr[l].qs[8];
                ধ্রুবক int64_t a2 = *(ধ্রুবক int64_t *)&a_ptr[l].qs[16];
                ধ্রুবক int64_t a3 = *(ধ্রুবক int64_t *)&a_ptr[l].qs[24];
                __asm__ __volatile__("" ::: "memory"); // prevent gcc from emitting fused vlse64, violating alignment constraints
                ধ্রুবক vint8m2_t lhs_0_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a0, vl / 4));
                ধ্রুবক vint8m2_t lhs_1_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a1, vl / 4));
                ধ্রুবক vint8m2_t lhs_2_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a2, vl / 4));
                ধ্রুবক vint8m2_t lhs_3_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a3, vl / 4));

                ধ্রুবক vint8m4_t rhs_raw_vec = __riscv_vle8_v_i8m4((ধ্রুবক int8_t *)b_ptr[l].qs, vl * 4);
                ধ্রুবক vint8m4_t rhs_vec_lo = __riscv_vsra_vx_i8m4(__riscv_vsll_vx_i8m4(rhs_raw_vec, 4, vl * 4), 4, vl * 4);
                ধ্রুবক vint8m4_t rhs_vec_hi = __riscv_vsra_vx_i8m4(rhs_raw_vec, 4, vl * 4);
                ধ্রুবক vint8m2_t rhs_vec_lo_0 = __riscv_vget_v_i8m4_i8m2(rhs_vec_lo, 0);
                ধ্রুবক vint8m2_t rhs_vec_lo_1 = __riscv_vget_v_i8m4_i8m2(rhs_vec_lo, 1);
                ধ্রুবক vint8m2_t rhs_vec_hi_0 = __riscv_vget_v_i8m4_i8m2(rhs_vec_hi, 0);
                ধ্রুবক vint8m2_t rhs_vec_hi_1 = __riscv_vget_v_i8m4_i8m2(rhs_vec_hi, 1);

                ধ্রুবক vint16m4_t sumi_lo_0 = __riscv_vwmul_vv_i16m4(rhs_vec_lo_0, lhs_0_8, vl * 2);
                ধ্রুবক vint16m4_t sumi_lo_1 = __riscv_vwmacc_vv_i16m4(sumi_lo_0, rhs_vec_lo_1, lhs_1_8, vl * 2);
                ধ্রুবক vint16m4_t sumi_hi_0 = __riscv_vwmacc_vv_i16m4(sumi_lo_1, rhs_vec_hi_0, lhs_2_8, vl * 2);
                ধ্রুবক vint16m4_t sumi_hi_m = __riscv_vwmacc_vv_i16m4(sumi_hi_0, rhs_vec_hi_1, lhs_3_8, vl * 2);

                ধ্রুবক vuint32m4_t sumi_i32 = __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vreinterpret_v_i16m4_i32m4(sumi_hi_m));
                ধ্রুবক vuint16m2_t sumi_h2_0 = __riscv_vnsrl_wx_u16m2(sumi_i32, 0, vl);
                ধ্রুবক vuint16m2_t sumi_h2_1 = __riscv_vnsrl_wx_u16m2(sumi_i32, 16, vl);
                ধ্রুবক vuint16m2_t sumi_h2 = __riscv_vadd_vv_u16m2(sumi_h2_0, sumi_h2_1, vl);
                ধ্রুবক vuint32m2_t sumi_h2_i32 = __riscv_vreinterpret_v_u16m2_u32m2(sumi_h2);
                ধ্রুবক vuint16m1_t sumi_h4_0 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 0, vl / 2);
                ধ্রুবক vuint16m1_t sumi_h4_1 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 16, vl / 2);
                ধ্রুবক vuint16m1_t sumi_h4 = __riscv_vadd_vv_u16m1(sumi_h4_0, sumi_h4_1, vl / 2);
                ধ্রুবক vuint32m1_t sumi_h4_i32 = __riscv_vreinterpret_v_u16m1_u32m1(sumi_h4);
                ধ্রুবক vint16mf2_t sumi_h8_0 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 0, vl / 4));
                ধ্রুবক vint16mf2_t sumi_h8_1 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 16, vl / 4));
                ধ্রুবক vint32m1_t sumi_h8 = __riscv_vwadd_vv_i32m1(sumi_h8_0, sumi_h8_1, vl / 4);
                ধ্রুবক vfloat32m1_t facc = __riscv_vfcvt_f_x_v_f32m1(sumi_h8, vl / 4);

                // vector version needs Zvfhmin extension
                ধ্রুবক দশমিক a_scale = GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                ধ্রুবক দশমিক b_scales[8] = {
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[0]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[1]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[2]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[3]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[4]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[5]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[6]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[7])
                };
                ধ্রুবক vfloat32m1_t b_scales_vec = __riscv_vle32_v_f32m1(b_scales, vl / 4);
                ধ্রুবক vfloat32m1_t tmp1 = __riscv_vfmul_vf_f32m1(facc, a_scale, vl / 4);
                sumf = __riscv_vfmacc_vv_f32m1(sumf, tmp1, b_scales_vec, vl / 4);
            }
            __riscv_vse32_v_f32m1(s + x * ncols_interleaved, sumf, vl / 4);
        }
        ফেরত;
    }

#endif
    ggml_gemv_q4_0_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

#যদি defined __riscv_zvfh
শূন্য ggml_gemv_q4_0_16x1_q8_0(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 16;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 1;

    প্রতিজ্ঞা (n % qk == 0);
    প্রতিজ্ঞা (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    ধ্রুবক block_q8_0 * a_ptr = (ধ্রুবক block_q8_0 *) vy;
    জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
        ধ্রুবক block_q4_0x16 * b_ptr = (ধ্রুবক block_q4_0x16 *) vx + (x * nb);

        // 1x16 Accumulator
        vfloat32m2_t sumf = __riscv_vfmv_v_f_f32m2(0.0f, 16);

        জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
            // 1x16 Integer Accumulator
            vint16m1_t sumi_0_lo_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
            vint16m1_t sumi_0_hi_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);

            // Accumulation loop.
            জন্য (পূর্ণসংখ্যা i = 0; i < QK4_0 / 2; i++) {
                // Load `b_ptr`.
                ধ্রুবক vint8mf2_t b_0_packed = __riscv_vle8_v_i8mf2((ধ্রুবক int8_t *)&b_ptr[l].qs[i * 16], 16);
                ধ্রুবক vint8mf2_t b_0_lo = __riscv_vsra_vx_i8mf2(__riscv_vsll_vx_i8mf2(b_0_packed, 4, 16), 4, 16);
                ধ্রুবক vint8mf2_t b_0_hi = __riscv_vsra_vx_i8mf2(b_0_packed, 4, 16);

                sumi_0_lo_16 = __riscv_vwmacc_vx_i16m1(sumi_0_lo_16, a_ptr[l].qs[i], b_0_lo, 16);
                sumi_0_hi_16 = __riscv_vwmacc_vx_i16m1(sumi_0_hi_16, a_ptr[l].qs[16 + i], b_0_hi, 16);
            }

            ধ্রুবক vint32m2_t sumi = __riscv_vwadd_vv_i32m2(sumi_0_lo_16, sumi_0_hi_16, 16);

            ধ্রুবক vfloat16m1_t b_d = __riscv_vle16_v_f16m1((ধ্রুবক _Float16 *)b_ptr[l].d, 16);
            ধ্রুবক vfloat32m2_t d_0 = __riscv_vfwmul_vf_f32m2(b_d, *(ধ্রুবক _Float16 *)&a_ptr[l].d, 16);

            sumf = __riscv_vfmacc_vv_f32m2(sumf, __riscv_vfcvt_f_x_v_f32m2(sumi, 16), d_0, 16);
        }

        __riscv_vse32_v_f32m2(s + x * 16, sumf, 16);
    }
}

শূন্য ggml_gemv_q4_K_16x1_q8_K(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK_K;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 16;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 1;

    প্রতিজ্ঞা (n % qk == 0);
    প্রতিজ্ঞা (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    ধ্রুবক block_q8_K * a_ptr = (ধ্রুবক block_q8_K *) vy;

    জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
        ধ্রুবক block_q4_Kx16 * b_ptr = (ধ্রুবক block_q4_Kx16 *) vx + (x * nb);

        // 1x16 Accumulator
        vfloat32m2_t sumf = __riscv_vfmv_v_f_f32m2(0.0f, 16);

        জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
            vint32m2_t sumi = __riscv_vmv_v_x_i32m2(0, 16);

            // Load `dmin`.
            ধ্রুবক vfloat32m2_t dmins_d = __riscv_vfmul_vf_f32m2(
                __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1((ধ্রুবক _Float16 *)b_ptr[l].dmin, 16), 16), a_ptr[l].d, 16);

            // We process 4 sub-blocks at once.
            জন্য (পূর্ণসংখ্যা j = 0; j < QK_K / 128; j++) {
                // Extract the scales and the mins.
                //
                // Low bits.
                vuint8m2_t scales_mins_lo = __riscv_vle8_v_u8m2(&b_ptr[l].scales[j * 64], 64);
                vuint8m2_t scales_lo = __riscv_vand_vx_u8m2(scales_mins_lo, 0x0F, 64);
                vuint8m2_t mins_lo = __riscv_vsrl_vx_u8m2(scales_mins_lo, 4, 64);

                // High bits.
                vuint8m2_t scales_mins_hi = __riscv_vle8_v_u8m2(&b_ptr[l].scales[128], 64);
                vuint8m2_t scales_hi;
                vuint8m2_t mins_hi;
                যদি (!j) {
                    scales_hi = __riscv_vsll_vx_u8m2(__riscv_vand_vx_u8m2(scales_mins_hi, 0x03, 64), 4, 64);
                    mins_hi = __riscv_vsll_vx_u8m2(__riscv_vand_vx_u8m2(scales_mins_hi, 0x0C, 64), 2, 64);
                } অন্যথা {
                    scales_hi = __riscv_vand_vx_u8m2(scales_mins_hi, 0x30, 64);
                    mins_hi = __riscv_vsrl_vx_u8m2(__riscv_vand_vx_u8m2(scales_mins_hi, 0xC0, 64), 2, 64);
                }
                vuint16m4_t scales = __riscv_vzext_vf2_u16m4(__riscv_vor_vv_u8m2(scales_hi, scales_lo, 64), 64);
                vint16m4_t mins = __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vzext_vf2_u16m4(__riscv_vor_vv_u8m2(mins_hi, mins_lo, 64), 64));

                // Reduce the mins and multiply with `dmin`.
                //
                // Correct in `sumf`.
                vint32m2_t bsums = __riscv_vmv_v_x_i32m2(0, 16);
                bsums = __riscv_vwmacc_vx_i32m2(bsums, a_ptr[l].bsums[j * 8] + a_ptr[l].bsums[j * 8 + 1], __riscv_vget_v_i16m4_i16m1(mins, 0), 16);
                bsums = __riscv_vwmacc_vx_i32m2(bsums, a_ptr[l].bsums[j * 8 + 2] + a_ptr[l].bsums[j * 8 + 3], __riscv_vget_v_i16m4_i16m1(mins, 1), 16);
                bsums = __riscv_vwmacc_vx_i32m2(bsums, a_ptr[l].bsums[j * 8 + 4] + a_ptr[l].bsums[j * 8 + 5], __riscv_vget_v_i16m4_i16m1(mins, 2), 16);
                bsums = __riscv_vwmacc_vx_i32m2(bsums, a_ptr[l].bsums[j * 8 + 6] + a_ptr[l].bsums[j * 8 + 7], __riscv_vget_v_i16m4_i16m1(mins, 3), 16);

                sumf = __riscv_vfsub_vv_f32m2(sumf, __riscv_vfmul_vv_f32m2(dmins_d, __riscv_vfcvt_f_x_v_f32m2(bsums, 16), 16), 16);

                // Accumulation for 2 sub-blocks.
                //
                // This might overflow, so we accumulate in two steps.
                //
                // Recheck.
                জন্য (পূর্ণসংখ্যা k = 0; k < 2; k++) {
                    vint16m1_t sumi_s_0_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
                    vint16m1_t sumi_s_1_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);

                    জন্য (পূর্ণসংখ্যা i = k * 16; i < k * 16 + QK4_0 / 2; i++) {
                        // Load `b_ptr`.
                        ধ্রুবক vuint8mf2_t b_0_packed = __riscv_vle8_v_u8mf2(&b_ptr[l].qs[j * 1024 + i * 16], 16);
                        ধ্রুবক vint8mf2_t b_s_0 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vand_vx_u8mf2(b_0_packed, 0xF, 16));
                        ধ্রুবক vint8mf2_t b_s_1 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vsrl_vx_u8mf2(b_0_packed, 4, 16));

                        sumi_s_0_16 = __riscv_vwmacc_vx_i16m1(sumi_s_0_16, a_ptr[l].qs[j * 128 + i], b_s_0, 16);
                        sumi_s_1_16 = __riscv_vwmacc_vx_i16m1(sumi_s_1_16, a_ptr[l].qs[j * 128 + 32 + i], b_s_1, 16);
                    }

                    sumi = __riscv_vwmacc_vv_i32m2(sumi,
                        __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(scales, 0)),
                        sumi_s_0_16, 16);
                    sumi = __riscv_vwmacc_vv_i32m2(sumi,
                        __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(scales, 1)),
                        sumi_s_1_16, 16);
                }
                // Accumulation for 2 sub-blocks.
                //
                // This might overflow, so we accumulate in two steps.
                //
                // Recheck.
                জন্য (পূর্ণসংখ্যা k = 0; k < 2; k++) {
                    vint16m1_t sumi_s_0_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
                    vint16m1_t sumi_s_1_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);

                    জন্য (পূর্ণসংখ্যা i = k * 16; i < k * 16 + QK4_0 / 2; i++) {
                        // Load `b_ptr`.
                        ধ্রুবক vuint8mf2_t b_0_packed = __riscv_vle8_v_u8mf2(&b_ptr[l].qs[j * 1024 + 512 + i * 16], 16);
                        ধ্রুবক vint8mf2_t b_s_0 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vand_vx_u8mf2(b_0_packed, 0xF, 16));
                        ধ্রুবক vint8mf2_t b_s_1 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vsrl_vx_u8mf2(b_0_packed, 4, 16));

                        sumi_s_0_16 = __riscv_vwmacc_vx_i16m1(sumi_s_0_16, a_ptr[l].qs[j * 128 + 64 + i], b_s_0, 16);
                        sumi_s_1_16 = __riscv_vwmacc_vx_i16m1(sumi_s_1_16, a_ptr[l].qs[j * 128 + 96 + i], b_s_1, 16);
                    }

                    sumi = __riscv_vwmacc_vv_i32m2(sumi,
                        __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(scales, 2)),
                        sumi_s_0_16, 16);
                    sumi = __riscv_vwmacc_vv_i32m2(sumi,
                        __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(scales, 3)),
                        sumi_s_1_16, 16);
                }
            }

            ধ্রুবক vfloat32m2_t b_d = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1((ধ্রুবক _Float16 *)&b_ptr[l].d[0], 16), 16);
            ধ্রুবক vfloat32m2_t d_0 = __riscv_vfmul_vf_f32m2(b_d, a_ptr[l].d, 16);

            sumf = __riscv_vfmacc_vv_f32m2(sumf, __riscv_vfcvt_f_x_v_f32m2(sumi, 16), d_0, 16);
        }

        __riscv_vse32_v_f32m2(s + x * 16, sumf, 16);
    }
}

শূন্য ggml_gemv_iq4_nl_16x1_q8_0(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 16;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 1;

    প্রতিজ্ঞা (n % qk == 0);
    প্রতিজ্ঞা (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    ধ্রুবক vint8mf2_t মানসমূহ = __riscv_vle8_v_i8mf2(kvalues_iq4nl, 16);
    ধ্রুবক block_q8_0 * a_ptr = (ধ্রুবক block_q8_0 *) vy;
    জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
        ধ্রুবক block_iq4_nlx16 * b_ptr = (ধ্রুবক block_iq4_nlx16 *) vx + (x * nb);

        // 1x16 Accumulator1
        vfloat32m2_t sumf = __riscv_vfmv_v_f_f32m2(0.0f, 16);

        জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
            // 1x16 integer accumulator
            vint32m2_t sumi = __riscv_vmv_v_x_i32m2(0.0f, 16);

            // Accumulation loop.
            জন্য (পূর্ণসংখ্যা i = 0; i < QK4_NL / 2; i++) {
                // Load `b_ptr`.
                ধ্রুবক vuint8mf2_t b_0_packed = __riscv_vle8_v_u8mf2((ধ্রুবক uint8_t *)&b_ptr[l].qs[i * 16], 16);
                ধ্রুবক vint8mf2_t b_0_lo = __riscv_vrgather_vv_i8mf2(মানসমূহ, __riscv_vand_vx_u8mf2(b_0_packed, 0xf, 16), 16);
                ধ্রুবক vint8mf2_t b_0_hi = __riscv_vrgather_vv_i8mf2(মানসমূহ, __riscv_vsrl_vx_u8mf2(b_0_packed, 4, 16), 16);
                // const vint16m1_t b_0_lo_16 = __riscv_vwcvt_x_x_v_i16m1(b_0_lo, 16);
                // const vint16m1_t b_0_hi_16 = __riscv_vwcvt_x_x_v_i16m1(b_0_hi, 16);

                ধ্রুবক vint16m1_t sumi_lo = __riscv_vwmul_vx_i16m1(b_0_lo, a_ptr[l].qs[i], 16);
                ধ্রুবক vint16m1_t sumi_hi = __riscv_vwmul_vx_i16m1(b_0_hi, a_ptr[l].qs[16 + i], 16);
                sumi = __riscv_vadd_vv_i32m2(sumi, __riscv_vwadd_vv_i32m2(sumi_lo, sumi_hi, 16), 16);
            }

            ধ্রুবক vfloat16m1_t b_d = __riscv_vle16_v_f16m1((ধ্রুবক _Float16 *)b_ptr[l].d, 16);
            ধ্রুবক vfloat32m2_t d_0 = __riscv_vfwmul_vf_f32m2(b_d, *(ধ্রুবক _Float16 *)&a_ptr[l].d, 16);

            sumf = __riscv_vfmacc_vv_f32m2(sumf, __riscv_vfcvt_f_x_v_f32m2(sumi, 16), d_0, 16);
        }

        __riscv_vse32_v_f32m2(s + x * 16, sumf, 16);
    }
}

শূন্য ggml_gemv_q8_0_16x1_q8_0(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 16;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 1;

    প্রতিজ্ঞা (n % qk == 0);
    প্রতিজ্ঞা (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);
    UNUSED(bs);

    ধ্রুবক block_q8_0 * a_ptr = (ধ্রুবক block_q8_0 *) vy;
    জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
        ধ্রুবক block_q8_0x16 * b_ptr = (ধ্রুবক block_q8_0x16 *) vx + (x * nb);

        // 1x16 Accumulator
        vfloat32m2_t sumf = __riscv_vfmv_v_f_f32m2(0.0f, 16);

        জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
            // 1x16 Integer Accumulator
            vint32m2_t sumi = __riscv_vmv_v_x_i32m2(0.0f, 16);

            // Accumulation loop.
            জন্য (পূর্ণসংখ্যা i = 0; i < QK8_0; i++) {
                // Load `b_ptr`.
                ধ্রুবক vint8mf2_t b_0 = __riscv_vle8_v_i8mf2((ধ্রুবক int8_t *)&b_ptr[l].qs[i * 16], 16);
                // const vint16m1_t b_0_16 = __riscv_vwcvt_x_x_v_i16m1(b_0, 16);

                sumi = __riscv_vwadd_wv_i32m2(sumi, __riscv_vwmul_vx_i16m1(b_0, a_ptr[l].qs[i], 16), 16);
            }

            ধ্রুবক vfloat16m1_t b_d = __riscv_vle16_v_f16m1((ধ্রুবক _Float16 *)b_ptr[l].d, 16);
            ধ্রুবক vfloat32m2_t d_0 = __riscv_vfwmul_vf_f32m2(b_d, *(ধ্রুবক _Float16 *)&a_ptr[l].d, 16);

            sumf = __riscv_vfmacc_vv_f32m2(sumf, __riscv_vfcvt_f_x_v_f32m2(sumi, 16), d_0, 16);
        }

        __riscv_vse32_v_f32m2(s + x * 16, sumf, 16);
    }
}

শূন্য ggml_gemv_q2_K_16x1_q8_K(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    প্রতিজ্ঞা(n % QK_K == 0);
    প্রতিজ্ঞা(nr == 1);
    প্রতিজ্ঞা(nc % 16 == 0);

    UNUSED(bs);

    ধ্রুবক পূর্ণসংখ্যা N_COLS_TILE = 16;
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_k_blocks = n / QK_K;

    ধ্রুবক size_t vl = __riscv_vsetvl_e32m2(N_COLS_TILE);
    জন্য (পূর্ণসংখ্যা col_tile = 0; col_tile < nc; col_tile += N_COLS_TILE) {

        ধ্রুবক block_q8_K* lhs_base_ptr = (ধ্রুবক block_q8_K*)vy;
        ধ্রুবক block_q2_Kx16* rhs_base_ptr = (ধ্রুবক block_q2_Kx16*)vx + (col_tile / N_COLS_TILE) * অঙ্ক_k_blocks;

        vfloat32m2_t v_sumf = __riscv_vfmv_v_f_f32m2(0.0f, vl);

        জন্য (পূর্ণসংখ্যা k_block = 0; k_block < অঙ্ক_k_blocks; ++k_block) {
            ধ্রুবক block_q8_K* lhs_current = &lhs_base_ptr[k_block];
            ধ্রুবক block_q2_Kx16* rhs_current = &rhs_base_ptr[k_block];

            // 1. Prepare Global Min Scales
            vfloat16m1_t v_g_min_f16 = __riscv_vle16_v_f16m1((ধ্রুবক _Float16*)rhs_current->dmin, vl);
            vfloat32m2_t v_g_min_base = __riscv_vfwcvt_f_f_v_f32m2(v_g_min_f16, vl);

            vfloat32m2_t v_g_min_final = __riscv_vfmul_vf_f32m2(v_g_min_base, lhs_current->d, vl);

            vint32m2_t v_isum = __riscv_vmv_v_x_i32m2(0, vl);

            ধ্রুবক uint8_t* rhs_qs_ptr = rhs_current->qs;
            ধ্রুবক uint8_t* rhs_sc_ptr = rhs_current->scales;
            ধ্রুবক int8_t*  lhs_qs_ptr = lhs_current->qs;

            // --- Phase Loop (4 phases x 64 elements) ---
            জন্য (পূর্ণসংখ্যা phase = 0; phase < 4; ++phase) {

                // A. Load Scales/Mins
                vuint16m1_t v_d_sb_0, v_d_sb_1, v_d_sb_2, v_d_sb_3;
                vuint16m1_t v_m_sb_0, v_m_sb_1, v_m_sb_2, v_m_sb_3;

                {
                    vuint8mf2_t v_raw;
                    // Sub-block 0
                    v_raw = __riscv_vle8_v_u8mf2(rhs_sc_ptr + 0, vl);
                    v_d_sb_0 = __riscv_vzext_vf2_u16m1(__riscv_vand_vx_u8mf2(v_raw, 0xF, vl), vl);
                    v_m_sb_0 = __riscv_vzext_vf2_u16m1(__riscv_vsrl_vx_u8mf2(v_raw, 4, vl), vl);

                    // Sub-block 1
                    v_raw = __riscv_vle8_v_u8mf2(rhs_sc_ptr + 16, vl);
                    v_d_sb_1 = __riscv_vzext_vf2_u16m1(__riscv_vand_vx_u8mf2(v_raw, 0xF, vl), vl);
                    v_m_sb_1 = __riscv_vzext_vf2_u16m1(__riscv_vsrl_vx_u8mf2(v_raw, 4, vl), vl);

                    // Sub-block 2
                    v_raw = __riscv_vle8_v_u8mf2(rhs_sc_ptr + 32, vl);
                    v_d_sb_2 = __riscv_vzext_vf2_u16m1(__riscv_vand_vx_u8mf2(v_raw, 0xF, vl), vl);
                    v_m_sb_2 = __riscv_vzext_vf2_u16m1(__riscv_vsrl_vx_u8mf2(v_raw, 4, vl), vl);

                    // Sub-block 3
                    v_raw = __riscv_vle8_v_u8mf2(rhs_sc_ptr + 48, vl);
                    v_d_sb_3 = __riscv_vzext_vf2_u16m1(__riscv_vand_vx_u8mf2(v_raw, 0xF, vl), vl);
                    v_m_sb_3 = __riscv_vzext_vf2_u16m1(__riscv_vsrl_vx_u8mf2(v_raw, 4, vl), vl);

                    rhs_sc_ptr += 64;
                }

                পূর্ণসংখ্যা base_k_phase = (phase < 2) ? (phase * 16) : (128 + (phase-2)*16);
                পূর্ণসংখ্যা k_offsets[4] = {0, 32, 64, 96};

                // B. Inner Dot Product Loop
                জন্য (পূর্ণসংখ্যা l = 0; l < 16; ++l) {
                    vuint8mf2_t v_rhs_তথ্য = __riscv_vle8_v_u8mf2(rhs_qs_ptr, vl);
                    rhs_qs_ptr += 16;

                    // Sub-block 0
                    {
                        vuint8mf2_t v_q2 = __riscv_vand_vx_u8mf2(v_rhs_তথ্য, 3, vl);
                        vint16m1_t v_w = __riscv_vmul_vv_i16m1(
                            __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(v_q2, vl)),
                            __riscv_vreinterpret_v_u16m1_i16m1(v_d_sb_0), vl);

                        int8_t q8 = lhs_qs_ptr[base_k_phase + k_offsets[0] + l];
                        v_isum = __riscv_vwmacc_vx_i32m2(v_isum, (int16_t)q8, v_w, vl);
                    }
                    // Sub-block 1
                    {
                        vuint8mf2_t v_q2 = __riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(v_rhs_তথ্য, 2, vl), 3, vl);
                        vint16m1_t v_w = __riscv_vmul_vv_i16m1(
                            __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(v_q2, vl)),
                            __riscv_vreinterpret_v_u16m1_i16m1(v_d_sb_1), vl);

                        int8_t q8 = lhs_qs_ptr[base_k_phase + k_offsets[1] + l];
                        v_isum = __riscv_vwmacc_vx_i32m2(v_isum, (int16_t)q8, v_w, vl);
                    }
                    // Sub-block 2
                    {
                        vuint8mf2_t v_q2 = __riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(v_rhs_তথ্য, 4, vl), 3, vl);
                        vint16m1_t v_w = __riscv_vmul_vv_i16m1(
                            __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(v_q2, vl)),
                            __riscv_vreinterpret_v_u16m1_i16m1(v_d_sb_2), vl);

                        int8_t q8 = lhs_qs_ptr[base_k_phase + k_offsets[2] + l];
                        v_isum = __riscv_vwmacc_vx_i32m2(v_isum, (int16_t)q8, v_w, vl);
                    }
                    // Sub-block 3
                    {
                        vuint8mf2_t v_q2 = __riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(v_rhs_তথ্য, 6, vl), 3, vl);
                        vint16m1_t v_w = __riscv_vmul_vv_i16m1(
                            __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(v_q2, vl)),
                            __riscv_vreinterpret_v_u16m1_i16m1(v_d_sb_3), vl);

                        int8_t q8 = lhs_qs_ptr[base_k_phase + k_offsets[3] + l];
                        v_isum = __riscv_vwmacc_vx_i32m2(v_isum, (int16_t)q8, v_w, vl);
                    }
                }

                // correction
                পূর্ণসংখ্যা sb_base_abs = base_k_phase / 16;

                // Sub-block 0
                {
                    পূর্ণসংখ্যা sb_idx = sb_base_abs + (k_offsets[0] / 16);
                    int16_t bsum = lhs_current->bsums[sb_idx];
                    vint16m1_t v_min = __riscv_vreinterpret_v_u16m1_i16m1(v_m_sb_0);
                    vint32m2_t v_c = __riscv_vwmul_vx_i32m2(v_min, bsum, vl);
                    vfloat32m2_t vf_c = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(v_c, vl), v_g_min_final, vl);
                    v_sumf = __riscv_vfsub_vv_f32m2(v_sumf, vf_c, vl);
                }
                // Sub-block 1
                {
                    পূর্ণসংখ্যা sb_idx = sb_base_abs + (k_offsets[1] / 16);
                    int16_t bsum = lhs_current->bsums[sb_idx];
                    vint16m1_t v_min = __riscv_vreinterpret_v_u16m1_i16m1(v_m_sb_1);
                    vint32m2_t v_c = __riscv_vwmul_vx_i32m2(v_min, bsum, vl);
                    vfloat32m2_t vf_c = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(v_c, vl), v_g_min_final, vl);
                    v_sumf = __riscv_vfsub_vv_f32m2(v_sumf, vf_c, vl);
                }
                // Sub-block 2
                {
                    পূর্ণসংখ্যা sb_idx = sb_base_abs + (k_offsets[2] / 16);
                    int16_t bsum = lhs_current->bsums[sb_idx];
                    vint16m1_t v_min = __riscv_vreinterpret_v_u16m1_i16m1(v_m_sb_2);
                    vint32m2_t v_c = __riscv_vwmul_vx_i32m2(v_min, bsum, vl);
                    vfloat32m2_t vf_c = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(v_c, vl), v_g_min_final, vl);
                    v_sumf = __riscv_vfsub_vv_f32m2(v_sumf, vf_c, vl);
                }
                // Sub-block 3
                {
                    পূর্ণসংখ্যা sb_idx = sb_base_abs + (k_offsets[3] / 16);
                    int16_t bsum = lhs_current->bsums[sb_idx];
                    vint16m1_t v_min = __riscv_vreinterpret_v_u16m1_i16m1(v_m_sb_3);
                    vint32m2_t v_c = __riscv_vwmul_vx_i32m2(v_min, bsum, vl);
                    vfloat32m2_t vf_c = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(v_c, vl), v_g_min_final, vl);
                    v_sumf = __riscv_vfsub_vv_f32m2(v_sumf, vf_c, vl);
                }

            } // End Phase Loop

            // Apply global Scales
            vfloat16m1_t v_g_all_f16 = __riscv_vle16_v_f16m1((ধ্রুবক _Float16*)rhs_current->d, vl);
            vfloat32m2_t v_g_all_base = __riscv_vfwcvt_f_f_v_f32m2(v_g_all_f16, vl);

            vfloat32m2_t v_g_all_final = __riscv_vfmul_vf_f32m2(v_g_all_base, lhs_current->d, vl);
            vfloat32m2_t v_যোগ = __riscv_vfcvt_f_x_v_f32m2(v_isum, vl);
            v_যোগ = __riscv_vfmul_vv_f32m2(v_যোগ, v_g_all_final, vl);
            v_sumf = __riscv_vfadd_vv_f32m2(v_sumf, v_যোগ, vl);

        } // End K-Block
        __riscv_vse32_v_f32m2(s + col_tile, v_sumf, vl);
    }
}
#endif

শূন্য ggml_gemm_q4_0_8x8_q8_0(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 8;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 8;

    প্রতিজ্ঞা (n % qk == 0);
    প্রতিজ্ঞা (nr % 4 == 0);
    প্রতিজ্ঞা (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#যদি defined __riscv_v
    যদি (__riscv_vlenb() >= QK4_0) {
        ধ্রুবক size_t vl = QK4_0;

        জন্য (পূর্ণসংখ্যা y = 0; y < nr / 4; y++) {
            ধ্রুবক block_q8_0x4 * a_ptr = (ধ্রুবক block_q8_0x4 *) vy + (y * nb);
            জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
                ধ্রুবক block_q4_0x8 * b_ptr = (ধ্রুবক block_q4_0x8 *) vx + (x * nb);
                vfloat32m1_t sumf0 = __riscv_vfmv_v_f_f32m1(0.0, vl / 4);
                vfloat32m1_t sumf1 = __riscv_vfmv_v_f_f32m1(0.0, vl / 4);
                vfloat32m1_t sumf2 = __riscv_vfmv_v_f_f32m1(0.0, vl / 4);
                vfloat32m1_t sumf3 = __riscv_vfmv_v_f_f32m1(0.0, vl / 4);
                জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
                    ধ্রুবক vint8m4_t rhs_raw_vec = __riscv_vle8_v_i8m4((ধ্রুবক int8_t *)b_ptr[l].qs, vl * 4);
                    ধ্রুবক vint8m4_t rhs_vec_lo = __riscv_vsra_vx_i8m4(__riscv_vsll_vx_i8m4(rhs_raw_vec, 4, vl * 4), 4, vl * 4);
                    ধ্রুবক vint8m4_t rhs_vec_hi = __riscv_vsra_vx_i8m4(rhs_raw_vec, 4, vl * 4);
                    ধ্রুবক vint8m2_t rhs_vec_lo_0 = __riscv_vget_v_i8m4_i8m2(rhs_vec_lo, 0);
                    ধ্রুবক vint8m2_t rhs_vec_lo_1 = __riscv_vget_v_i8m4_i8m2(rhs_vec_lo, 1);
                    ধ্রুবক vint8m2_t rhs_vec_hi_0 = __riscv_vget_v_i8m4_i8m2(rhs_vec_hi, 0);
                    ধ্রুবক vint8m2_t rhs_vec_hi_1 = __riscv_vget_v_i8m4_i8m2(rhs_vec_hi, 1);

                    // vector version needs Zvfhmin extension
                    ধ্রুবক দশমিক a_scales[4] = {
                        GGML_CPU_FP16_TO_FP32(a_ptr[l].d[0]),
                        GGML_CPU_FP16_TO_FP32(a_ptr[l].d[1]),
                        GGML_CPU_FP16_TO_FP32(a_ptr[l].d[2]),
                        GGML_CPU_FP16_TO_FP32(a_ptr[l].d[3])
                    };
                    ধ্রুবক দশমিক b_scales[8] = {
                        GGML_CPU_FP16_TO_FP32(b_ptr[l].d[0]),
                        GGML_CPU_FP16_TO_FP32(b_ptr[l].d[1]),
                        GGML_CPU_FP16_TO_FP32(b_ptr[l].d[2]),
                        GGML_CPU_FP16_TO_FP32(b_ptr[l].d[3]),
                        GGML_CPU_FP16_TO_FP32(b_ptr[l].d[4]),
                        GGML_CPU_FP16_TO_FP32(b_ptr[l].d[5]),
                        GGML_CPU_FP16_TO_FP32(b_ptr[l].d[6]),
                        GGML_CPU_FP16_TO_FP32(b_ptr[l].d[7])
                    };
                    ধ্রুবক vfloat32m1_t b_scales_vec = __riscv_vle32_v_f32m1(b_scales, vl / 4);

                    ধ্রুবক int64_t A0 = *(ধ্রুবক int64_t *)&a_ptr[l].qs[0];
                    ধ্রুবক int64_t A4 = *(ধ্রুবক int64_t *)&a_ptr[l].qs[32];
                    ধ্রুবক int64_t A8 = *(ধ্রুবক int64_t *)&a_ptr[l].qs[64];
                    ধ্রুবক int64_t Ac = *(ধ্রুবক int64_t *)&a_ptr[l].qs[96];
                    __asm__ __volatile__("" ::: "memory"); // prevent gcc from emitting fused vlse64, violating alignment
                    vint16m4_t sumi_l0;
                    {
                        ধ্রুবক vint8m2_t lhs_0_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A0, vl / 4));
                        ধ্রুবক vint8m2_t lhs_1_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A4, vl / 4));
                        ধ্রুবক vint8m2_t lhs_2_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A8, vl / 4));
                        ধ্রুবক vint8m2_t lhs_3_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(Ac, vl / 4));
                        ধ্রুবক vint16m4_t sumi_lo_0 = __riscv_vwmul_vv_i16m4(rhs_vec_lo_0, lhs_0_8, vl * 2);
                        ধ্রুবক vint16m4_t sumi_lo_1 = __riscv_vwmacc_vv_i16m4(sumi_lo_0, rhs_vec_lo_1, lhs_1_8, vl * 2);
                        ধ্রুবক vint16m4_t sumi_hi_0 = __riscv_vwmacc_vv_i16m4(sumi_lo_1, rhs_vec_hi_0, lhs_2_8, vl * 2);
                        ধ্রুবক vint16m4_t sumi_hi_m = __riscv_vwmacc_vv_i16m4(sumi_hi_0, rhs_vec_hi_1, lhs_3_8, vl * 2);

                        sumi_l0 = sumi_hi_m;
                    }

                    {
                        ধ্রুবক vuint32m4_t sumi_i32 = __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vreinterpret_v_i16m4_i32m4(sumi_l0));
                        ধ্রুবক vuint16m2_t sumi_h2_0 = __riscv_vnsrl_wx_u16m2(sumi_i32, 0, vl);
                        ধ্রুবক vuint16m2_t sumi_h2_1 = __riscv_vnsrl_wx_u16m2(sumi_i32, 16, vl);
                        ধ্রুবক vuint16m2_t sumi_h2 = __riscv_vadd_vv_u16m2(sumi_h2_0, sumi_h2_1, vl);
                        ধ্রুবক vuint32m2_t sumi_h2_i32 = __riscv_vreinterpret_v_u16m2_u32m2(sumi_h2);
                        ধ্রুবক vuint16m1_t sumi_h4_0 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 0, vl / 2);
                        ধ্রুবক vuint16m1_t sumi_h4_1 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 16, vl / 2);
                        ধ্রুবক vuint16m1_t sumi_h4 = __riscv_vadd_vv_u16m1(sumi_h4_0, sumi_h4_1, vl / 2);
                        ধ্রুবক vuint32m1_t sumi_h4_i32 = __riscv_vreinterpret_v_u16m1_u32m1(sumi_h4);
                        ধ্রুবক vint16mf2_t sumi_h8_0 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 0, vl / 4));
                        ধ্রুবক vint16mf2_t sumi_h8_1 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 16, vl / 4));
                        ধ্রুবক vint32m1_t sumi_h8 = __riscv_vwadd_vv_i32m1(sumi_h8_0, sumi_h8_1, vl / 4);
                        ধ্রুবক vfloat32m1_t facc = __riscv_vfcvt_f_x_v_f32m1(sumi_h8, vl / 4);

                        ধ্রুবক vfloat32m1_t tmp1 = __riscv_vfmul_vf_f32m1(facc, a_scales[0], vl / 4);
                        sumf0 = __riscv_vfmacc_vv_f32m1(sumf0, tmp1, b_scales_vec, vl / 4);
                    }

                    ধ্রুবক int64_t A1 = *(ধ্রুবক int64_t *)&a_ptr[l].qs[8];
                    ধ্রুবক int64_t A5 = *(ধ্রুবক int64_t *)&a_ptr[l].qs[40];
                    ধ্রুবক int64_t A9 = *(ধ্রুবক int64_t *)&a_ptr[l].qs[72];
                    ধ্রুবক int64_t Ad = *(ধ্রুবক int64_t *)&a_ptr[l].qs[104];
                    __asm__ __volatile__("" ::: "memory"); // prevent gcc from emitting fused vlse64, violating alignment
                    vint16m4_t sumi_l1;
                    {
                        ধ্রুবক vint8m2_t lhs_0_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A1, vl / 4));
                        ধ্রুবক vint8m2_t lhs_1_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A5, vl / 4));
                        ধ্রুবক vint8m2_t lhs_2_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A9, vl / 4));
                        ধ্রুবক vint8m2_t lhs_3_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(Ad, vl / 4));
                        ধ্রুবক vint16m4_t sumi_lo_0 = __riscv_vwmul_vv_i16m4(rhs_vec_lo_0, lhs_0_8, vl * 2);
                        ধ্রুবক vint16m4_t sumi_lo_1 = __riscv_vwmacc_vv_i16m4(sumi_lo_0, rhs_vec_lo_1, lhs_1_8, vl * 2);
                        ধ্রুবক vint16m4_t sumi_hi_0 = __riscv_vwmacc_vv_i16m4(sumi_lo_1, rhs_vec_hi_0, lhs_2_8, vl * 2);
                        ধ্রুবক vint16m4_t sumi_hi_m = __riscv_vwmacc_vv_i16m4(sumi_hi_0, rhs_vec_hi_1, lhs_3_8, vl * 2);

                        sumi_l1 = sumi_hi_m;
                    }

                    {
                        ধ্রুবক vuint32m4_t sumi_i32 = __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vreinterpret_v_i16m4_i32m4(sumi_l1));
                        ধ্রুবক vuint16m2_t sumi_h2_0 = __riscv_vnsrl_wx_u16m2(sumi_i32, 0, vl);
                        ধ্রুবক vuint16m2_t sumi_h2_1 = __riscv_vnsrl_wx_u16m2(sumi_i32, 16, vl);
                        ধ্রুবক vuint16m2_t sumi_h2 = __riscv_vadd_vv_u16m2(sumi_h2_0, sumi_h2_1, vl);
                        ধ্রুবক vuint32m2_t sumi_h2_i32 = __riscv_vreinterpret_v_u16m2_u32m2(sumi_h2);
                        ধ্রুবক vuint16m1_t sumi_h4_0 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 0, vl / 2);
                        ধ্রুবক vuint16m1_t sumi_h4_1 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 16, vl / 2);
                        ধ্রুবক vuint16m1_t sumi_h4 = __riscv_vadd_vv_u16m1(sumi_h4_0, sumi_h4_1, vl / 2);
                        ধ্রুবক vuint32m1_t sumi_h4_i32 = __riscv_vreinterpret_v_u16m1_u32m1(sumi_h4);
                        ধ্রুবক vint16mf2_t sumi_h8_0 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 0, vl / 4));
                        ধ্রুবক vint16mf2_t sumi_h8_1 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 16, vl / 4));
                        ধ্রুবক vint32m1_t sumi_h8 = __riscv_vwadd_vv_i32m1(sumi_h8_0, sumi_h8_1, vl / 4);
                        ধ্রুবক vfloat32m1_t facc = __riscv_vfcvt_f_x_v_f32m1(sumi_h8, vl / 4);

                        ধ্রুবক vfloat32m1_t tmp1 = __riscv_vfmul_vf_f32m1(facc, a_scales[1], vl / 4);
                        sumf1 = __riscv_vfmacc_vv_f32m1(sumf1, tmp1, b_scales_vec, vl / 4);
                    }

                    ধ্রুবক int64_t A2 = *(ধ্রুবক int64_t *)&a_ptr[l].qs[16];
                    ধ্রুবক int64_t A6 = *(ধ্রুবক int64_t *)&a_ptr[l].qs[48];
                    ধ্রুবক int64_t Aa = *(ধ্রুবক int64_t *)&a_ptr[l].qs[80];
                    ধ্রুবক int64_t Ae = *(ধ্রুবক int64_t *)&a_ptr[l].qs[112];
                    __asm__ __volatile__("" ::: "memory"); // prevent gcc from emitting fused vlse64, violating alignment
                    vint16m4_t sumi_l2;
                    {
                        ধ্রুবক vint8m2_t lhs_0_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A2, vl / 4));
                        ধ্রুবক vint8m2_t lhs_1_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A6, vl / 4));
                        ধ্রুবক vint8m2_t lhs_2_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(Aa, vl / 4));
                        ধ্রুবক vint8m2_t lhs_3_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(Ae, vl / 4));
                        ধ্রুবক vint16m4_t sumi_lo_0 = __riscv_vwmul_vv_i16m4(rhs_vec_lo_0, lhs_0_8, vl * 2);
                        ধ্রুবক vint16m4_t sumi_lo_1 = __riscv_vwmacc_vv_i16m4(sumi_lo_0, rhs_vec_lo_1, lhs_1_8, vl * 2);
                        ধ্রুবক vint16m4_t sumi_hi_0 = __riscv_vwmacc_vv_i16m4(sumi_lo_1, rhs_vec_hi_0, lhs_2_8, vl * 2);
                        ধ্রুবক vint16m4_t sumi_hi_m = __riscv_vwmacc_vv_i16m4(sumi_hi_0, rhs_vec_hi_1, lhs_3_8, vl * 2);

                        sumi_l2 = sumi_hi_m;
                    }

                    {
                        ধ্রুবক vuint32m4_t sumi_i32 = __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vreinterpret_v_i16m4_i32m4(sumi_l2));
                        ধ্রুবক vuint16m2_t sumi_h2_0 = __riscv_vnsrl_wx_u16m2(sumi_i32, 0, vl);
                        ধ্রুবক vuint16m2_t sumi_h2_1 = __riscv_vnsrl_wx_u16m2(sumi_i32, 16, vl);
                        ধ্রুবক vuint16m2_t sumi_h2 = __riscv_vadd_vv_u16m2(sumi_h2_0, sumi_h2_1, vl);
                        ধ্রুবক vuint32m2_t sumi_h2_i32 = __riscv_vreinterpret_v_u16m2_u32m2(sumi_h2);
                        ধ্রুবক vuint16m1_t sumi_h4_0 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 0, vl / 2);
                        ধ্রুবক vuint16m1_t sumi_h4_1 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 16, vl / 2);
                        ধ্রুবক vuint16m1_t sumi_h4 = __riscv_vadd_vv_u16m1(sumi_h4_0, sumi_h4_1, vl / 2);
                        ধ্রুবক vuint32m1_t sumi_h4_i32 = __riscv_vreinterpret_v_u16m1_u32m1(sumi_h4);
                        ধ্রুবক vint16mf2_t sumi_h8_0 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 0, vl / 4));
                        ধ্রুবক vint16mf2_t sumi_h8_1 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 16, vl / 4));
                        ধ্রুবক vint32m1_t sumi_h8 = __riscv_vwadd_vv_i32m1(sumi_h8_0, sumi_h8_1, vl / 4);
                        ধ্রুবক vfloat32m1_t facc = __riscv_vfcvt_f_x_v_f32m1(sumi_h8, vl / 4);

                        ধ্রুবক vfloat32m1_t tmp1 = __riscv_vfmul_vf_f32m1(facc, a_scales[2], vl / 4);
                        sumf2 = __riscv_vfmacc_vv_f32m1(sumf2, tmp1, b_scales_vec, vl / 4);
                    }

                    ধ্রুবক int64_t A3 = *(ধ্রুবক int64_t *)&a_ptr[l].qs[24];
                    ধ্রুবক int64_t A7 = *(ধ্রুবক int64_t *)&a_ptr[l].qs[56];
                    ধ্রুবক int64_t Ab = *(ধ্রুবক int64_t *)&a_ptr[l].qs[88];
                    ধ্রুবক int64_t Af = *(ধ্রুবক int64_t *)&a_ptr[l].qs[120];
                    __asm__ __volatile__("" ::: "memory"); // prevent gcc from emitting fused vlse64, violating alignment
                    vint16m4_t sumi_l3;
                    {
                        ধ্রুবক vint8m2_t lhs_0_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A3, vl / 4));
                        ধ্রুবক vint8m2_t lhs_1_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A7, vl / 4));
                        ধ্রুবক vint8m2_t lhs_2_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(Ab, vl / 4));
                        ধ্রুবক vint8m2_t lhs_3_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(Af, vl / 4));
                        ধ্রুবক vint16m4_t sumi_lo_0 = __riscv_vwmul_vv_i16m4(rhs_vec_lo_0, lhs_0_8, vl * 2);
                        ধ্রুবক vint16m4_t sumi_lo_1 = __riscv_vwmacc_vv_i16m4(sumi_lo_0, rhs_vec_lo_1, lhs_1_8, vl * 2);
                        ধ্রুবক vint16m4_t sumi_hi_0 = __riscv_vwmacc_vv_i16m4(sumi_lo_1, rhs_vec_hi_0, lhs_2_8, vl * 2);
                        ধ্রুবক vint16m4_t sumi_hi_m = __riscv_vwmacc_vv_i16m4(sumi_hi_0, rhs_vec_hi_1, lhs_3_8, vl * 2);

                        sumi_l3 = sumi_hi_m;
                    }

                    {
                        ধ্রুবক vuint32m4_t sumi_i32 = __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vreinterpret_v_i16m4_i32m4(sumi_l3));
                        ধ্রুবক vuint16m2_t sumi_h2_0 = __riscv_vnsrl_wx_u16m2(sumi_i32, 0, vl);
                        ধ্রুবক vuint16m2_t sumi_h2_1 = __riscv_vnsrl_wx_u16m2(sumi_i32, 16, vl);
                        ধ্রুবক vuint16m2_t sumi_h2 = __riscv_vadd_vv_u16m2(sumi_h2_0, sumi_h2_1, vl);
                        ধ্রুবক vuint32m2_t sumi_h2_i32 = __riscv_vreinterpret_v_u16m2_u32m2(sumi_h2);
                        ধ্রুবক vuint16m1_t sumi_h4_0 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 0, vl / 2);
                        ধ্রুবক vuint16m1_t sumi_h4_1 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 16, vl / 2);
                        ধ্রুবক vuint16m1_t sumi_h4 = __riscv_vadd_vv_u16m1(sumi_h4_0, sumi_h4_1, vl / 2);
                        ধ্রুবক vuint32m1_t sumi_h4_i32 = __riscv_vreinterpret_v_u16m1_u32m1(sumi_h4);
                        ধ্রুবক vint16mf2_t sumi_h8_0 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 0, vl / 4));
                        ধ্রুবক vint16mf2_t sumi_h8_1 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 16, vl / 4));
                        ধ্রুবক vint32m1_t sumi_h8 = __riscv_vwadd_vv_i32m1(sumi_h8_0, sumi_h8_1, vl / 4);
                        ধ্রুবক vfloat32m1_t facc = __riscv_vfcvt_f_x_v_f32m1(sumi_h8, vl / 4);

                        ধ্রুবক vfloat32m1_t tmp1 = __riscv_vfmul_vf_f32m1(facc, a_scales[3], vl / 4);
                        sumf3 = __riscv_vfmacc_vv_f32m1(sumf3, tmp1, b_scales_vec, vl / 4);
                    }
                }
                __riscv_vse32_v_f32m1(&s[(y * 4 + 0) * bs + x * ncols_interleaved], sumf0, vl / 4);
                __riscv_vse32_v_f32m1(&s[(y * 4 + 1) * bs + x * ncols_interleaved], sumf1, vl / 4);
                __riscv_vse32_v_f32m1(&s[(y * 4 + 2) * bs + x * ncols_interleaved], sumf2, vl / 4);
                __riscv_vse32_v_f32m1(&s[(y * 4 + 3) * bs + x * ncols_interleaved], sumf3, vl / 4);
            }
        }

        ফেরত;
    }

#endif
    ggml_gemm_q4_0_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

#যদি defined __riscv_zvfh
শূন্য ggml_gemm_q4_0_16x1_q8_0(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 16;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 1;

    প্রতিজ্ঞা (n % qk == 0);
    প্রতিজ্ঞা (nr % 4 == 0);
    প্রতিজ্ঞা (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    জন্য (পূর্ণসংখ্যা y = 0; y < nr / 4; y++) {
        ধ্রুবক block_q8_0x4 * a_ptr = (ধ্রুবক block_q8_0x4 *) vy + (y * nb);
        জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
            ধ্রুবক block_q4_0x16 * b_ptr = (ধ্রুবক block_q4_0x16 *) vx + (x * nb);

            // 4x16 Accumulators
            vfloat32m2_t sumf_0 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
            vfloat32m2_t sumf_1 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
            vfloat32m2_t sumf_2 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
            vfloat32m2_t sumf_3 = __riscv_vfmv_v_f_f32m2(0.0f, 16);

            জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
                // 4x16 integer accumulators
                vint16m1_t sumi_0_lo_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
                vint16m1_t sumi_1_lo_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
                vint16m1_t sumi_2_lo_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
                vint16m1_t sumi_3_lo_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
                vint16m1_t sumi_0_hi_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
                vint16m1_t sumi_1_hi_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
                vint16m1_t sumi_2_hi_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
                vint16m1_t sumi_3_hi_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);

                // Accumulation loop.
                জন্য (পূর্ণসংখ্যা i = 0; i < QK4_0 / 2; i++) {
                    // Load `b_ptr`.
                    ধ্রুবক vint8mf2_t b_0_packed = __riscv_vle8_v_i8mf2((ধ্রুবক int8_t *)&b_ptr[l].qs[i * 16], 16);
                    ধ্রুবক vint8mf2_t b_0_lo = __riscv_vsra_vx_i8mf2(__riscv_vsll_vx_i8mf2(b_0_packed, 4, 16), 4, 16);
                    ধ্রুবক vint8mf2_t b_0_hi = __riscv_vsra_vx_i8mf2(b_0_packed, 4, 16);

                    sumi_0_lo_16 = __riscv_vwmacc_vx_i16m1(sumi_0_lo_16, a_ptr[l].qs[i * 4], b_0_lo, 16);
                    sumi_1_lo_16 = __riscv_vwmacc_vx_i16m1(sumi_1_lo_16, a_ptr[l].qs[i * 4 + 1], b_0_lo, 16);
                    sumi_2_lo_16 = __riscv_vwmacc_vx_i16m1(sumi_2_lo_16, a_ptr[l].qs[i * 4 + 2], b_0_lo, 16);
                    sumi_3_lo_16 = __riscv_vwmacc_vx_i16m1(sumi_3_lo_16, a_ptr[l].qs[i * 4 + 3], b_0_lo, 16);

                    sumi_0_hi_16 = __riscv_vwmacc_vx_i16m1(sumi_0_hi_16, a_ptr[l].qs[64 + i * 4], b_0_hi, 16);
                    sumi_1_hi_16 = __riscv_vwmacc_vx_i16m1(sumi_1_hi_16, a_ptr[l].qs[64 + i * 4 + 1], b_0_hi, 16);
                    sumi_2_hi_16 = __riscv_vwmacc_vx_i16m1(sumi_2_hi_16, a_ptr[l].qs[64 + i * 4 + 2], b_0_hi, 16);
                    sumi_3_hi_16 = __riscv_vwmacc_vx_i16m1(sumi_3_hi_16, a_ptr[l].qs[64 + i * 4 + 3], b_0_hi, 16);
                }

                // Do the final accumulation in i32 to prevent overflow.
                ধ্রুবক vint32m2_t sumi_0 = __riscv_vwadd_vv_i32m2(sumi_0_lo_16, sumi_0_hi_16, 16);
                ধ্রুবক vint32m2_t sumi_1 = __riscv_vwadd_vv_i32m2(sumi_1_lo_16, sumi_1_hi_16, 16);
                ধ্রুবক vint32m2_t sumi_2 = __riscv_vwadd_vv_i32m2(sumi_2_lo_16, sumi_2_hi_16, 16);
                ধ্রুবক vint32m2_t sumi_3 = __riscv_vwadd_vv_i32m2(sumi_3_lo_16, sumi_3_hi_16, 16);

                ধ্রুবক vfloat16m1_t b_d = __riscv_vle16_v_f16m1((ধ্রুবক _Float16 *)b_ptr[l].d, 16);
                ধ্রুবক vfloat32m2_t d_0 = __riscv_vfwmul_vf_f32m2(b_d, *(ধ্রুবক _Float16 *)&a_ptr[l].d[0], 16);
                ধ্রুবক vfloat32m2_t d_1 = __riscv_vfwmul_vf_f32m2(b_d, *(ধ্রুবক _Float16 *)&a_ptr[l].d[1], 16);
                ধ্রুবক vfloat32m2_t d_2 = __riscv_vfwmul_vf_f32m2(b_d, *(ধ্রুবক _Float16 *)&a_ptr[l].d[2], 16);
                ধ্রুবক vfloat32m2_t d_3 = __riscv_vfwmul_vf_f32m2(b_d, *(ধ্রুবক _Float16 *)&a_ptr[l].d[3], 16);

                sumf_0 = __riscv_vfmacc_vv_f32m2(sumf_0, __riscv_vfcvt_f_x_v_f32m2(sumi_0, 16), d_0, 16);
                sumf_1 = __riscv_vfmacc_vv_f32m2(sumf_1, __riscv_vfcvt_f_x_v_f32m2(sumi_1, 16), d_1, 16);
                sumf_2 = __riscv_vfmacc_vv_f32m2(sumf_2, __riscv_vfcvt_f_x_v_f32m2(sumi_2, 16), d_2, 16);
                sumf_3 = __riscv_vfmacc_vv_f32m2(sumf_3, __riscv_vfcvt_f_x_v_f32m2(sumi_3, 16), d_3, 16);
            }

            __riscv_vse32_v_f32m2(s + (y * 4 + 0) * bs + x * 16, sumf_0, 16);
            __riscv_vse32_v_f32m2(s + (y * 4 + 1) * bs + x * 16, sumf_1, 16);
            __riscv_vse32_v_f32m2(s + (y * 4 + 2) * bs + x * 16, sumf_2, 16);
            __riscv_vse32_v_f32m2(s + (y * 4 + 3) * bs + x * 16, sumf_3, 16);
        }
    }
}

শূন্য ggml_gemm_q4_K_16x1_q8_K(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK_K;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 16;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 1;

    প্রতিজ্ঞা (n % qk == 0);
    প্রতিজ্ঞা (nr % 4 == 0);
    প্রতিজ্ঞা (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    জন্য (পূর্ণসংখ্যা y = 0; y < nr / 4; y++) {
        ধ্রুবক block_q8_Kx4 * a_ptr = (ধ্রুবক block_q8_Kx4 *) vy + (y * nb);
        জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
            ধ্রুবক block_q4_Kx16 * b_ptr = (ধ্রুবক block_q4_Kx16 *) vx + (x * nb);

            // 4x16 Accumulators
            vfloat32m2_t sumf_0 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
            vfloat32m2_t sumf_1 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
            vfloat32m2_t sumf_2 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
            vfloat32m2_t sumf_3 = __riscv_vfmv_v_f_f32m2(0.0f, 16);

            জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
                vint32m2_t sumi_0 = __riscv_vmv_v_x_i32m2(0, 16);
                vint32m2_t sumi_1 = __riscv_vmv_v_x_i32m2(0, 16);
                vint32m2_t sumi_2 = __riscv_vmv_v_x_i32m2(0, 16);
                vint32m2_t sumi_3 = __riscv_vmv_v_x_i32m2(0, 16);

                // Load `dmin`.
                ধ্রুবক vfloat32m2_t dmins = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1((ধ্রুবক _Float16 *)b_ptr[l].dmin, 16), 16);

                // We process 4 sub-blocks at once.
                জন্য (পূর্ণসংখ্যা j = 0; j < QK_K / 128; j++) {
                    // Extract the scales and the mins.
                    //
                    // Low bits.
                    vuint8m2_t scales_mins_lo = __riscv_vle8_v_u8m2(&b_ptr[l].scales[j * 64], 64);
                    vuint8m2_t scales_lo = __riscv_vand_vx_u8m2(scales_mins_lo, 0x0F, 64);
                    vuint8m2_t mins_lo = __riscv_vsrl_vx_u8m2(scales_mins_lo, 4, 64);

                    // High bits.
                    vuint8m2_t scales_mins_hi = __riscv_vle8_v_u8m2(&b_ptr[l].scales[128], 64);
                    vuint8m2_t scales_hi;
                    vuint8m2_t mins_hi;
                    যদি (!j) {
                        scales_hi = __riscv_vsll_vx_u8m2(__riscv_vand_vx_u8m2(scales_mins_hi, 0x03, 64), 4, 64);
                        mins_hi = __riscv_vsll_vx_u8m2(__riscv_vand_vx_u8m2(scales_mins_hi, 0x0C, 64), 2, 64);
                    } অন্যথা {
                        scales_hi = __riscv_vand_vx_u8m2(scales_mins_hi, 0x30, 64);
                        mins_hi = __riscv_vsrl_vx_u8m2(__riscv_vand_vx_u8m2(scales_mins_hi, 0xC0, 64), 2, 64);
                    }
                    vuint16m4_t scales = __riscv_vzext_vf2_u16m4(__riscv_vor_vv_u8m2(scales_hi, scales_lo, 64), 64);
                    vint16m4_t mins = __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vzext_vf2_u16m4(__riscv_vor_vv_u8m2(mins_hi, mins_lo, 64), 64));

                    // Reduce the mins and multiply with `dmin`.
                    //
                    // Correct in `sumf`.
                    vint32m2_t bsums_0 = __riscv_vmv_v_x_i32m2(0, 16);
                    vint32m2_t bsums_1 = __riscv_vmv_v_x_i32m2(0, 16);
                    vint32m2_t bsums_2 = __riscv_vmv_v_x_i32m2(0, 16);
                    vint32m2_t bsums_3 = __riscv_vmv_v_x_i32m2(0, 16);

                    bsums_0 = __riscv_vwmacc_vx_i32m2(bsums_0,
                                a_ptr[l].bsums[j * 32] + a_ptr[l].bsums[j * 32 + 4],
                                __riscv_vget_v_i16m4_i16m1(mins, 0), 16);
                    bsums_1 = __riscv_vwmacc_vx_i32m2(bsums_1,
                                a_ptr[l].bsums[j * 32 + 1] + a_ptr[l].bsums[j * 32 + 5],
                                __riscv_vget_v_i16m4_i16m1(mins, 0), 16);
                    bsums_2 = __riscv_vwmacc_vx_i32m2(bsums_2,
                                a_ptr[l].bsums[j * 32 + 2] + a_ptr[l].bsums[j * 32 + 6],
                                __riscv_vget_v_i16m4_i16m1(mins, 0), 16);
                    bsums_3 = __riscv_vwmacc_vx_i32m2(bsums_3,
                                a_ptr[l].bsums[j * 32 + 3] + a_ptr[l].bsums[j * 32 + 7],
                                __riscv_vget_v_i16m4_i16m1(mins, 0), 16);
                    bsums_0 = __riscv_vwmacc_vx_i32m2(bsums_0,
                                a_ptr[l].bsums[j * 32 + 8] + a_ptr[l].bsums[j * 32 + 8 + 4],
                                __riscv_vget_v_i16m4_i16m1(mins, 1), 16);
                    bsums_1 = __riscv_vwmacc_vx_i32m2(bsums_1,
                                a_ptr[l].bsums[j * 32 + 8 + 1] + a_ptr[l].bsums[j * 32 + 8 + 5],
                                __riscv_vget_v_i16m4_i16m1(mins, 1), 16);
                    bsums_2 = __riscv_vwmacc_vx_i32m2(bsums_2,
                                a_ptr[l].bsums[j * 32 + 8 + 2] + a_ptr[l].bsums[j * 32 + 8 + 6],
                                __riscv_vget_v_i16m4_i16m1(mins, 1), 16);
                    bsums_3 = __riscv_vwmacc_vx_i32m2(bsums_3,
                                a_ptr[l].bsums[j * 32 + 8 + 3] + a_ptr[l].bsums[j * 32 + 8 + 7],
                                __riscv_vget_v_i16m4_i16m1(mins, 1), 16);
                    bsums_0 = __riscv_vwmacc_vx_i32m2(bsums_0,
                                a_ptr[l].bsums[j * 32 + 16] + a_ptr[l].bsums[j * 32 + 16 + 4],
                                __riscv_vget_v_i16m4_i16m1(mins, 2), 16);
                    bsums_1 = __riscv_vwmacc_vx_i32m2(bsums_1,
                                a_ptr[l].bsums[j * 32 + 16 + 1] + a_ptr[l].bsums[j * 32 + 16 + 5],
                                __riscv_vget_v_i16m4_i16m1(mins, 2), 16);
                    bsums_2 = __riscv_vwmacc_vx_i32m2(bsums_2,
                                a_ptr[l].bsums[j * 32 + 16 + 2] + a_ptr[l].bsums[j * 32 + 16 + 6],
                                __riscv_vget_v_i16m4_i16m1(mins, 2), 16);
                    bsums_3 = __riscv_vwmacc_vx_i32m2(bsums_3,
                                a_ptr[l].bsums[j * 32 + 16 + 3] + a_ptr[l].bsums[j * 32 + 16 + 7],
                                __riscv_vget_v_i16m4_i16m1(mins, 2), 16);
                    bsums_0 = __riscv_vwmacc_vx_i32m2(bsums_0,
                                a_ptr[l].bsums[j * 32 + 24 + 0] + a_ptr[l].bsums[j * 32 + 24 + 4],
                                __riscv_vget_v_i16m4_i16m1(mins, 3), 16);
                    bsums_1 = __riscv_vwmacc_vx_i32m2(bsums_1,
                                a_ptr[l].bsums[j * 32 + 24 + 1] + a_ptr[l].bsums[j * 32 + 24 + 5],
                                __riscv_vget_v_i16m4_i16m1(mins, 3), 16);
                    bsums_2 = __riscv_vwmacc_vx_i32m2(bsums_2,
                                a_ptr[l].bsums[j * 32 + 24 + 2] + a_ptr[l].bsums[j * 32 + 24 + 6],
                                __riscv_vget_v_i16m4_i16m1(mins, 3), 16);
                    bsums_3 = __riscv_vwmacc_vx_i32m2(bsums_3,
                                a_ptr[l].bsums[j * 32 + 24 + 3] + a_ptr[l].bsums[j * 32 + 24 + 7],
                                __riscv_vget_v_i16m4_i16m1(mins, 3), 16);

                    ধ্রুবক vfloat32m2_t dmins_d_0 = __riscv_vfmul_vf_f32m2(dmins, a_ptr[l].d[0], 16);
                    ধ্রুবক vfloat32m2_t dmins_d_1 = __riscv_vfmul_vf_f32m2(dmins, a_ptr[l].d[1], 16);
                    ধ্রুবক vfloat32m2_t dmins_d_2 = __riscv_vfmul_vf_f32m2(dmins, a_ptr[l].d[2], 16);
                    ধ্রুবক vfloat32m2_t dmins_d_3 = __riscv_vfmul_vf_f32m2(dmins, a_ptr[l].d[3], 16);

                    sumf_0 = __riscv_vfsub_vv_f32m2(sumf_0, __riscv_vfmul_vv_f32m2(dmins_d_0, __riscv_vfcvt_f_x_v_f32m2(bsums_0, 16), 16), 16);
                    sumf_1 = __riscv_vfsub_vv_f32m2(sumf_1, __riscv_vfmul_vv_f32m2(dmins_d_1, __riscv_vfcvt_f_x_v_f32m2(bsums_1, 16), 16), 16);
                    sumf_2 = __riscv_vfsub_vv_f32m2(sumf_2, __riscv_vfmul_vv_f32m2(dmins_d_2, __riscv_vfcvt_f_x_v_f32m2(bsums_2, 16), 16), 16);
                    sumf_3 = __riscv_vfsub_vv_f32m2(sumf_3, __riscv_vfmul_vv_f32m2(dmins_d_3, __riscv_vfcvt_f_x_v_f32m2(bsums_3, 16), 16), 16);


                    // Accumulation for 2 sub-blocks.
                    //
                    // This might overflow, so we accumulate in two steps.
                    //
                    // Recheck.
                    জন্য (পূর্ণসংখ্যা k = 0; k < 2; k++) {
                        // 4x16 integer accumulators
                        vint16m1_t sumi_0_s_0_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
                        vint16m1_t sumi_1_s_0_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
                        vint16m1_t sumi_2_s_0_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
                        vint16m1_t sumi_3_s_0_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
                        vint16m1_t sumi_0_s_1_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
                        vint16m1_t sumi_1_s_1_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
                        vint16m1_t sumi_2_s_1_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
                        vint16m1_t sumi_3_s_1_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);

                        জন্য (পূর্ণসংখ্যা i = k * 16; i < k * 16 + QK4_0 / 2; i++) {
                            // Load `b_ptr`.
                            ধ্রুবক vuint8mf2_t b_0_packed = __riscv_vle8_v_u8mf2(&b_ptr[l].qs[j * 1024 + i * 16], 16);
                            ধ্রুবক vint8mf2_t b_s_0 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vand_vx_u8mf2(b_0_packed, 0xF, 16));
                            ধ্রুবক vint8mf2_t b_s_1 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vsrl_vx_u8mf2(b_0_packed, 4, 16));

                            sumi_0_s_0_16 = __riscv_vwmacc_vx_i16m1(sumi_0_s_0_16, a_ptr[l].qs[j * 512 + i * 4], b_s_0, 16);
                            sumi_1_s_0_16 = __riscv_vwmacc_vx_i16m1(sumi_1_s_0_16, a_ptr[l].qs[j * 512 + i * 4 + 1], b_s_0, 16);
                            sumi_2_s_0_16 = __riscv_vwmacc_vx_i16m1(sumi_2_s_0_16, a_ptr[l].qs[j * 512 + i * 4 + 2], b_s_0, 16);
                            sumi_3_s_0_16 = __riscv_vwmacc_vx_i16m1(sumi_3_s_0_16, a_ptr[l].qs[j * 512 + i * 4 + 3], b_s_0, 16);

                            sumi_0_s_1_16 = __riscv_vwmacc_vx_i16m1(sumi_0_s_1_16, a_ptr[l].qs[j * 512 + 128 + i * 4], b_s_1, 16);
                            sumi_1_s_1_16 = __riscv_vwmacc_vx_i16m1(sumi_1_s_1_16, a_ptr[l].qs[j * 512 + 128 + i * 4 + 1], b_s_1, 16);
                            sumi_2_s_1_16 = __riscv_vwmacc_vx_i16m1(sumi_2_s_1_16, a_ptr[l].qs[j * 512 + 128 + i * 4 + 2], b_s_1, 16);
                            sumi_3_s_1_16 = __riscv_vwmacc_vx_i16m1(sumi_3_s_1_16, a_ptr[l].qs[j * 512 + 128 + i * 4 + 3], b_s_1, 16);
                        }

                        sumi_0 = __riscv_vwmacc_vv_i32m2(sumi_0,
                                    __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(scales, 0)),
                                    sumi_0_s_0_16, 16);
                        sumi_0 = __riscv_vwmacc_vv_i32m2(sumi_0,
                                    __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(scales, 1)),
                                    sumi_0_s_1_16, 16);
                        sumi_1 = __riscv_vwmacc_vv_i32m2(sumi_1,
                                    __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(scales, 0)),
                                    sumi_1_s_0_16, 16);
                        sumi_1 = __riscv_vwmacc_vv_i32m2(sumi_1,
                                    __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(scales, 1)),
                                    sumi_1_s_1_16, 16);
                        sumi_2 = __riscv_vwmacc_vv_i32m2(sumi_2,
                                    __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(scales, 0)),
                                    sumi_2_s_0_16, 16);
                        sumi_2 = __riscv_vwmacc_vv_i32m2(sumi_2,
                                    __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(scales, 1)),
                                    sumi_2_s_1_16, 16);
                        sumi_3 = __riscv_vwmacc_vv_i32m2(sumi_3,
                                    __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(scales, 0)),
                                    sumi_3_s_0_16, 16);
                        sumi_3 = __riscv_vwmacc_vv_i32m2(sumi_3,
                                    __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(scales, 1)),
                                    sumi_3_s_1_16, 16);
                    }
                    // Accumulation for 2 sub-blocks.
                    //
                    // This might overflow, so we accumulate in two steps.
                    //
                    // Recheck.
                    জন্য (পূর্ণসংখ্যা k = 0; k < 2; k++) {
                        // 4x16 integer accumulators
                        vint16m1_t sumi_0_s_0_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
                        vint16m1_t sumi_1_s_0_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
                        vint16m1_t sumi_2_s_0_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
                        vint16m1_t sumi_3_s_0_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
                        vint16m1_t sumi_0_s_1_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
                        vint16m1_t sumi_1_s_1_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
                        vint16m1_t sumi_2_s_1_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
                        vint16m1_t sumi_3_s_1_16 = __riscv_vmv_v_x_i16m1(0.0f, 16);

                        জন্য (পূর্ণসংখ্যা i = k * 16; i < k * 16 + QK4_0 / 2; i++) {
                            // Load `b_ptr`.
                            ধ্রুবক vuint8mf2_t b_0_packed = __riscv_vle8_v_u8mf2(&b_ptr[l].qs[j * 1024 + 512 + i * 16], 16);
                            ধ্রুবক vint8mf2_t b_s_0 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vand_vx_u8mf2(b_0_packed, 0xF, 16));
                            ধ্রুবক vint8mf2_t b_s_1 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vsrl_vx_u8mf2(b_0_packed, 4, 16));

                            sumi_0_s_0_16 = __riscv_vwmacc_vx_i16m1(sumi_0_s_0_16, a_ptr[l].qs[j * 512 + 256 + i * 4], b_s_0, 16);
                            sumi_1_s_0_16 = __riscv_vwmacc_vx_i16m1(sumi_1_s_0_16, a_ptr[l].qs[j * 512 + 256 + i * 4 + 1], b_s_0, 16);
                            sumi_2_s_0_16 = __riscv_vwmacc_vx_i16m1(sumi_2_s_0_16, a_ptr[l].qs[j * 512 + 256 + i * 4 + 2], b_s_0, 16);
                            sumi_3_s_0_16 = __riscv_vwmacc_vx_i16m1(sumi_3_s_0_16, a_ptr[l].qs[j * 512 + 256 + i * 4 + 3], b_s_0, 16);

                            sumi_0_s_1_16 = __riscv_vwmacc_vx_i16m1(sumi_0_s_1_16, a_ptr[l].qs[j * 512 + 384 + i * 4], b_s_1, 16);
                            sumi_1_s_1_16 = __riscv_vwmacc_vx_i16m1(sumi_1_s_1_16, a_ptr[l].qs[j * 512 + 384 + i * 4 + 1], b_s_1, 16);
                            sumi_2_s_1_16 = __riscv_vwmacc_vx_i16m1(sumi_2_s_1_16, a_ptr[l].qs[j * 512 + 384 + i * 4 + 2], b_s_1, 16);
                            sumi_3_s_1_16 = __riscv_vwmacc_vx_i16m1(sumi_3_s_1_16, a_ptr[l].qs[j * 512 + 384 + i * 4 + 3], b_s_1, 16);
                        }

                        sumi_0 = __riscv_vwmacc_vv_i32m2(sumi_0,
                                    __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(scales, 2)),
                                    sumi_0_s_0_16, 16);
                        sumi_0 = __riscv_vwmacc_vv_i32m2(sumi_0,
                                    __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(scales, 3)),
                                    sumi_0_s_1_16, 16);
                        sumi_1 = __riscv_vwmacc_vv_i32m2(sumi_1,
                                    __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(scales, 2)),
                                    sumi_1_s_0_16, 16);
                        sumi_1 = __riscv_vwmacc_vv_i32m2(sumi_1,
                                    __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(scales, 3)),
                                    sumi_1_s_1_16, 16);
                        sumi_2 = __riscv_vwmacc_vv_i32m2(sumi_2,
                                    __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(scales, 2)),
                                    sumi_2_s_0_16, 16);
                        sumi_2 = __riscv_vwmacc_vv_i32m2(sumi_2,
                                    __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(scales, 3)),
                                    sumi_2_s_1_16, 16);
                        sumi_3 = __riscv_vwmacc_vv_i32m2(sumi_3,
                                    __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(scales, 2)),
                                    sumi_3_s_0_16, 16);
                        sumi_3 = __riscv_vwmacc_vv_i32m2(sumi_3,
                                    __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(scales, 3)),
                                    sumi_3_s_1_16, 16);
                    }
                }

                ধ্রুবক vfloat32m2_t b_d = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1((ধ্রুবক _Float16 *)b_ptr[l].d, 16), 16);
                ধ্রুবক vfloat32m2_t d_0 = __riscv_vfmul_vf_f32m2(b_d, a_ptr[l].d[0], 16);
                ধ্রুবক vfloat32m2_t d_1 = __riscv_vfmul_vf_f32m2(b_d, a_ptr[l].d[1], 16);
                ধ্রুবক vfloat32m2_t d_2 = __riscv_vfmul_vf_f32m2(b_d, a_ptr[l].d[2], 16);
                ধ্রুবক vfloat32m2_t d_3 = __riscv_vfmul_vf_f32m2(b_d, a_ptr[l].d[3], 16);

                sumf_0 = __riscv_vfmacc_vv_f32m2(sumf_0, __riscv_vfcvt_f_x_v_f32m2(sumi_0, 16), d_0, 16);
                sumf_1 = __riscv_vfmacc_vv_f32m2(sumf_1, __riscv_vfcvt_f_x_v_f32m2(sumi_1, 16), d_1, 16);
                sumf_2 = __riscv_vfmacc_vv_f32m2(sumf_2, __riscv_vfcvt_f_x_v_f32m2(sumi_2, 16), d_2, 16);
                sumf_3 = __riscv_vfmacc_vv_f32m2(sumf_3, __riscv_vfcvt_f_x_v_f32m2(sumi_3, 16), d_3, 16);
            }

            __riscv_vse32_v_f32m2(s + (y * 4 + 0) * bs + x * 16, sumf_0, 16);
            __riscv_vse32_v_f32m2(s + (y * 4 + 1) * bs + x * 16, sumf_1, 16);
            __riscv_vse32_v_f32m2(s + (y * 4 + 2) * bs + x * 16, sumf_2, 16);
            __riscv_vse32_v_f32m2(s + (y * 4 + 3) * bs + x * 16, sumf_3, 16);
        }
    }
}

শূন্য ggml_gemm_iq4_nl_16x1_q8_0(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 16;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 1;

    প্রতিজ্ঞা (n % qk == 0);
    প্রতিজ্ঞা (nr % 4 == 0);
    প্রতিজ্ঞা (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    ধ্রুবক vint8mf2_t মানসমূহ = __riscv_vle8_v_i8mf2(kvalues_iq4nl, 16);
    জন্য (পূর্ণসংখ্যা y = 0; y < nr / 4; y++) {
        ধ্রুবক block_q8_0x4 * a_ptr = (ধ্রুবক block_q8_0x4 *) vy + (y * nb);
        জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
            ধ্রুবক block_iq4_nlx16 * b_ptr = (ধ্রুবক block_iq4_nlx16 *) vx + (x * nb);

            // 4x16 Accumulators
            vfloat32m2_t sumf_0 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
            vfloat32m2_t sumf_1 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
            vfloat32m2_t sumf_2 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
            vfloat32m2_t sumf_3 = __riscv_vfmv_v_f_f32m2(0.0f, 16);

            জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
                // 4x16 integer accumulators
                vint32m2_t sumi_0 = __riscv_vmv_v_x_i32m2(0.0f, 16);
                vint32m2_t sumi_1 = __riscv_vmv_v_x_i32m2(0.0f, 16);
                vint32m2_t sumi_2 = __riscv_vmv_v_x_i32m2(0.0f, 16);
                vint32m2_t sumi_3 = __riscv_vmv_v_x_i32m2(0.0f, 16);

                // Accumulation loop.
                জন্য (পূর্ণসংখ্যা i = 0; i < QK4_NL / 2; i++) {
                    // Load `b_ptr`.
                    ধ্রুবক vuint8mf2_t b_0_packed = __riscv_vle8_v_u8mf2((ধ্রুবক uint8_t *)&b_ptr[l].qs[i * 16], 16);
                    ধ্রুবক vint8mf2_t b_0_lo = __riscv_vrgather_vv_i8mf2(মানসমূহ, __riscv_vand_vx_u8mf2(b_0_packed, 0xf, 16), 16);
                    ধ্রুবক vint8mf2_t b_0_hi = __riscv_vrgather_vv_i8mf2(মানসমূহ, __riscv_vsrl_vx_u8mf2(b_0_packed, 4, 16), 16);
                    // const vint16m1_t b_0_lo_16 = __riscv_vwcvt_x_x_v_i16m1(b_0_lo, 16);
                    // const vint16m1_t b_0_hi_16 = __riscv_vwcvt_x_x_v_i16m1(b_0_hi, 16);

                    ধ্রুবক vint16m1_t sumi_0_lo = __riscv_vwmul_vx_i16m1(b_0_lo, a_ptr[l].qs[i * 4], 16);
                    ধ্রুবক vint16m1_t sumi_1_lo = __riscv_vwmul_vx_i16m1(b_0_lo, a_ptr[l].qs[i * 4 + 1], 16);
                    ধ্রুবক vint16m1_t sumi_2_lo = __riscv_vwmul_vx_i16m1(b_0_lo, a_ptr[l].qs[i * 4 + 2], 16);
                    ধ্রুবক vint16m1_t sumi_3_lo = __riscv_vwmul_vx_i16m1(b_0_lo, a_ptr[l].qs[i * 4 + 3], 16);

                    ধ্রুবক vint16m1_t sumi_0_hi = __riscv_vwmul_vx_i16m1(b_0_hi, a_ptr[l].qs[64 + i * 4], 16);
                    ধ্রুবক vint16m1_t sumi_1_hi = __riscv_vwmul_vx_i16m1(b_0_hi, a_ptr[l].qs[64 + i * 4 + 1], 16);
                    ধ্রুবক vint16m1_t sumi_2_hi = __riscv_vwmul_vx_i16m1(b_0_hi, a_ptr[l].qs[64 + i * 4 + 2], 16);
                    ধ্রুবক vint16m1_t sumi_3_hi = __riscv_vwmul_vx_i16m1(b_0_hi, a_ptr[l].qs[64 + i * 4 + 3], 16);

                    sumi_0 = __riscv_vadd_vv_i32m2(sumi_0, __riscv_vwadd_vv_i32m2(sumi_0_lo, sumi_0_hi, 16), 16);
                    sumi_1 = __riscv_vadd_vv_i32m2(sumi_1, __riscv_vwadd_vv_i32m2(sumi_1_lo, sumi_1_hi, 16), 16);
                    sumi_2 = __riscv_vadd_vv_i32m2(sumi_2, __riscv_vwadd_vv_i32m2(sumi_2_lo, sumi_2_hi, 16), 16);
                    sumi_3 = __riscv_vadd_vv_i32m2(sumi_3, __riscv_vwadd_vv_i32m2(sumi_3_lo, sumi_3_hi, 16), 16);
                }

                ধ্রুবক vfloat16m1_t b_d = __riscv_vle16_v_f16m1((ধ্রুবক _Float16 *)b_ptr[l].d, 16);
                ধ্রুবক vfloat32m2_t d_0 = __riscv_vfwmul_vf_f32m2(b_d, *(ধ্রুবক _Float16 *)&a_ptr[l].d[0], 16);
                ধ্রুবক vfloat32m2_t d_1 = __riscv_vfwmul_vf_f32m2(b_d, *(ধ্রুবক _Float16 *)&a_ptr[l].d[1], 16);
                ধ্রুবক vfloat32m2_t d_2 = __riscv_vfwmul_vf_f32m2(b_d, *(ধ্রুবক _Float16 *)&a_ptr[l].d[2], 16);
                ধ্রুবক vfloat32m2_t d_3 = __riscv_vfwmul_vf_f32m2(b_d, *(ধ্রুবক _Float16 *)&a_ptr[l].d[3], 16);

                sumf_0 = __riscv_vfmacc_vv_f32m2(sumf_0, __riscv_vfcvt_f_x_v_f32m2(sumi_0, 16), d_0, 16);
                sumf_1 = __riscv_vfmacc_vv_f32m2(sumf_1, __riscv_vfcvt_f_x_v_f32m2(sumi_1, 16), d_1, 16);
                sumf_2 = __riscv_vfmacc_vv_f32m2(sumf_2, __riscv_vfcvt_f_x_v_f32m2(sumi_2, 16), d_2, 16);
                sumf_3 = __riscv_vfmacc_vv_f32m2(sumf_3, __riscv_vfcvt_f_x_v_f32m2(sumi_3, 16), d_3, 16);
            }

            __riscv_vse32_v_f32m2(s + (y * 4 + 0) * bs + x * 16, sumf_0, 16);
            __riscv_vse32_v_f32m2(s + (y * 4 + 1) * bs + x * 16, sumf_1, 16);
            __riscv_vse32_v_f32m2(s + (y * 4 + 2) * bs + x * 16, sumf_2, 16);
            __riscv_vse32_v_f32m2(s + (y * 4 + 3) * bs + x * 16, sumf_3, 16);
        }
    }
}

শূন্য ggml_gemm_q8_0_16x1_q8_0(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 16;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 1;

    প্রতিজ্ঞা (n % qk == 0);
    প্রতিজ্ঞা (nr % 4 == 0);
    প্রতিজ্ঞা (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    জন্য (পূর্ণসংখ্যা y = 0; y < nr / 4; y++) {
        ধ্রুবক block_q8_0x4 * a_ptr = (ধ্রুবক block_q8_0x4 *) vy + (y * nb);
        জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
            ধ্রুবক block_q8_0x16 * b_ptr = (ধ্রুবক block_q8_0x16 *) vx + (x * nb);

            // 4x16 Accumulators
            vfloat32m2_t sumf_0 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
            vfloat32m2_t sumf_1 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
            vfloat32m2_t sumf_2 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
            vfloat32m2_t sumf_3 = __riscv_vfmv_v_f_f32m2(0.0f, 16);

            জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
                // 4x16 Integer Accumulators
                vint32m2_t sumi_0 = __riscv_vmv_v_x_i32m2(0.0f, 16);
                vint32m2_t sumi_1 = __riscv_vmv_v_x_i32m2(0.0f, 16);
                vint32m2_t sumi_2 = __riscv_vmv_v_x_i32m2(0.0f, 16);
                vint32m2_t sumi_3 = __riscv_vmv_v_x_i32m2(0.0f, 16);

                // Accumulation loop.
                জন্য (পূর্ণসংখ্যা i = 0; i < QK8_0; i++) {
                    // Load `b_ptr`.
                    ধ্রুবক vint8mf2_t b_0 = __riscv_vle8_v_i8mf2((ধ্রুবক int8_t *)&b_ptr[l].qs[i * 16], 16);
                    // const vint16m1_t b_0_16 = __riscv_vwcvt_x_x_v_i16m1(b_0, 16);

                    sumi_0 = __riscv_vwadd_wv_i32m2(sumi_0, __riscv_vwmul_vx_i16m1(b_0, a_ptr[l].qs[i * 4 + 0], 16), 16);
                    sumi_1 = __riscv_vwadd_wv_i32m2(sumi_1, __riscv_vwmul_vx_i16m1(b_0, a_ptr[l].qs[i * 4 + 1], 16), 16);
                    sumi_2 = __riscv_vwadd_wv_i32m2(sumi_2, __riscv_vwmul_vx_i16m1(b_0, a_ptr[l].qs[i * 4 + 2], 16), 16);
                    sumi_3 = __riscv_vwadd_wv_i32m2(sumi_3, __riscv_vwmul_vx_i16m1(b_0, a_ptr[l].qs[i * 4 + 3], 16), 16);
                }

                ধ্রুবক vfloat16m1_t b_d = __riscv_vle16_v_f16m1((ধ্রুবক _Float16 *)b_ptr[l].d, 16);
                ধ্রুবক vfloat32m2_t d_0 = __riscv_vfwmul_vf_f32m2(b_d, *(ধ্রুবক _Float16 *)&a_ptr[l].d[0], 16);
                ধ্রুবক vfloat32m2_t d_1 = __riscv_vfwmul_vf_f32m2(b_d, *(ধ্রুবক _Float16 *)&a_ptr[l].d[1], 16);
                ধ্রুবক vfloat32m2_t d_2 = __riscv_vfwmul_vf_f32m2(b_d, *(ধ্রুবক _Float16 *)&a_ptr[l].d[2], 16);
                ধ্রুবক vfloat32m2_t d_3 = __riscv_vfwmul_vf_f32m2(b_d, *(ধ্রুবক _Float16 *)&a_ptr[l].d[3], 16);

                sumf_0 = __riscv_vfmacc_vv_f32m2(sumf_0, __riscv_vfcvt_f_x_v_f32m2(sumi_0, 16), d_0, 16);
                sumf_1 = __riscv_vfmacc_vv_f32m2(sumf_1, __riscv_vfcvt_f_x_v_f32m2(sumi_1, 16), d_1, 16);
                sumf_2 = __riscv_vfmacc_vv_f32m2(sumf_2, __riscv_vfcvt_f_x_v_f32m2(sumi_2, 16), d_2, 16);
                sumf_3 = __riscv_vfmacc_vv_f32m2(sumf_3, __riscv_vfcvt_f_x_v_f32m2(sumi_3, 16), d_3, 16);
            }

            __riscv_vse32_v_f32m2(s + (y * 4 + 0) * bs + x * 16, sumf_0, 16);
            __riscv_vse32_v_f32m2(s + (y * 4 + 1) * bs + x * 16, sumf_1, 16);
            __riscv_vse32_v_f32m2(s + (y * 4 + 2) * bs + x * 16, sumf_2, 16);
            __riscv_vse32_v_f32m2(s + (y * 4 + 3) * bs + x * 16, sumf_3, 16);
        }
    }
}

শূন্য ggml_gemm_q2_K_16x1_q8_K(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    প্রতিজ্ঞা(n % QK_K == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_k_blocks = n / QK_K;
    ধ্রুবক পূর্ণসংখ্যা N_ROWS_TILE = 4;
    ধ্রুবক পূর্ণসংখ্যা N_COLS_TILE = 16;
    প্রতিজ্ঞা(nr % N_ROWS_TILE == 0);
    প্রতিজ্ঞা(nc % N_COLS_TILE == 0);

    ধ্রুবক size_t vl = __riscv_vsetvl_e32m2(N_COLS_TILE);
    // --- Tiling Loops ---
#pragma GCC unroll 1
    জন্য (পূর্ণসংখ্যা row_tile = 0; row_tile < nr; row_tile += N_ROWS_TILE) {
#pragma GCC unroll 1
        জন্য (পূর্ণসংখ্যা col_tile = 0; col_tile < nc; col_tile += N_COLS_TILE) {
            // Base Pointers
            ধ্রুবক block_q8_Kx4* lhs_base_ptr = (ধ্রুবক block_q8_Kx4*)vy + (row_tile / N_ROWS_TILE) * অঙ্ক_k_blocks;
            ধ্রুবক block_q2_Kx16* rhs_base_ptr = (ধ্রুবক block_q2_Kx16*)vx + (col_tile / N_COLS_TILE) * অঙ্ক_k_blocks;

            // Persistent Float Accumulators
            vfloat32m2_t v_sumf_0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t v_sumf_1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t v_sumf_2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t v_sumf_3 = __riscv_vfmv_v_f_f32m2(0.0f, vl);

            // --- Super-Block Loop (K=0..255) ---
#pragma GCC unroll 1
            জন্য (পূর্ণসংখ্যা k_block = 0; k_block < অঙ্ক_k_blocks; ++k_block) {
                ধ্রুবক block_q8_Kx4* lhs_current = &lhs_base_ptr[k_block];
                ধ্রুবক block_q2_Kx16* rhs_current = &rhs_base_ptr[k_block];

                // 1. Load Global Min Scales (Keep as F16/LMUL=1 to save registers)
                vfloat16m1_t v_g_min_f16 = __riscv_vle16_v_f16m1((ধ্রুবক _Float16*)rhs_current->dmin, vl);
                vfloat32m2_t v_g_min_base = __riscv_vfwcvt_f_f_v_f32m2(v_g_min_f16, vl);

                // 2. Initialize Integer Accumulators
                vint32m2_t v_isum_0 = __riscv_vmv_v_x_i32m2(0, vl);
                vint32m2_t v_isum_1 = __riscv_vmv_v_x_i32m2(0, vl);
                vint32m2_t v_isum_2 = __riscv_vmv_v_x_i32m2(0, vl);
                vint32m2_t v_isum_3 = __riscv_vmv_v_x_i32m2(0, vl);

                ধ্রুবক uint8_t* rhs_qs_ptr = rhs_current->qs;
                ধ্রুবক uint8_t* rhs_sc_ptr = rhs_current->scales;
                ধ্রুবক int8_t*  lhs_qs_ptr = lhs_current->qs;

                // --- Phase Loop (4 phases x 64 elements) ---
#pragma GCC unroll 1
                জন্য (পূর্ণসংখ্যা phase = 0; phase < 4; ++phase) {

                    // A. Load Scales/Mins for the 4 interleaved sub-blocks
                    vuint16m1_t v_d_sb_0, v_d_sb_1, v_d_sb_2, v_d_sb_3;
                    vuint16m1_t v_m_sb_0, v_m_sb_1, v_m_sb_2, v_m_sb_3;

                    // Unrolled Load Logic
                    {
                        vuint8mf2_t v_raw;
                        // Sub-block 0
                        v_raw = __riscv_vle8_v_u8mf2(rhs_sc_ptr + 0, vl);
                        v_d_sb_0 = __riscv_vzext_vf2_u16m1(__riscv_vand_vx_u8mf2(v_raw, 0xF, vl), vl);
                        v_m_sb_0 = __riscv_vzext_vf2_u16m1(__riscv_vsrl_vx_u8mf2(v_raw, 4, vl), vl);

                        // Sub-block 1
                        v_raw = __riscv_vle8_v_u8mf2(rhs_sc_ptr + 16, vl);
                        v_d_sb_1 = __riscv_vzext_vf2_u16m1(__riscv_vand_vx_u8mf2(v_raw, 0xF, vl), vl);
                        v_m_sb_1 = __riscv_vzext_vf2_u16m1(__riscv_vsrl_vx_u8mf2(v_raw, 4, vl), vl);

                        // Sub-block 2
                        v_raw = __riscv_vle8_v_u8mf2(rhs_sc_ptr + 32, vl);
                        v_d_sb_2 = __riscv_vzext_vf2_u16m1(__riscv_vand_vx_u8mf2(v_raw, 0xF, vl), vl);
                        v_m_sb_2 = __riscv_vzext_vf2_u16m1(__riscv_vsrl_vx_u8mf2(v_raw, 4, vl), vl);

                        // Sub-block 3
                        v_raw = __riscv_vle8_v_u8mf2(rhs_sc_ptr + 48, vl);
                        v_d_sb_3 = __riscv_vzext_vf2_u16m1(__riscv_vand_vx_u8mf2(v_raw, 0xF, vl), vl);
                        v_m_sb_3 = __riscv_vzext_vf2_u16m1(__riscv_vsrl_vx_u8mf2(v_raw, 4, vl), vl);

                        rhs_sc_ptr += 64;
                    }

                    পূর্ণসংখ্যা base_k_phase = (phase < 2) ? (phase * 16) : (128 + (phase-2)*16);
                    পূর্ণসংখ্যা k_offsets[4] = {0, 32, 64, 96};

                    // B. Inner Dot Product Loop
#pragma GCC unroll 1
                    জন্য (পূর্ণসংখ্যা l = 0; l < 16; ++l) {
                        vuint8mf2_t v_rhs_তথ্য = __riscv_vle8_v_u8mf2(rhs_qs_ptr, vl);
                        rhs_qs_ptr += 16;

                        // Unroll over 4 sub-blocks (0, 1, 2, 3 relative to phase)

                        // --- Sub-block 0 ---
                        {
                            vuint8mf2_t v_q2 = __riscv_vand_vx_u8mf2(v_rhs_তথ্য, 3, vl);
                            vint16m1_t v_w = __riscv_vmul_vv_i16m1(
                                __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(v_q2, vl)),
                                __riscv_vreinterpret_v_u16m1_i16m1(v_d_sb_0), vl);

                            ধ্রুবক int8_t* q8 = &lhs_qs_ptr[(base_k_phase + k_offsets[0] + l) * 4];
                            v_isum_0 = __riscv_vwmacc_vx_i32m2(v_isum_0, (int16_t)q8[0], v_w, vl);
                            v_isum_1 = __riscv_vwmacc_vx_i32m2(v_isum_1, (int16_t)q8[1], v_w, vl);
                            v_isum_2 = __riscv_vwmacc_vx_i32m2(v_isum_2, (int16_t)q8[2], v_w, vl);
                            v_isum_3 = __riscv_vwmacc_vx_i32m2(v_isum_3, (int16_t)q8[3], v_w, vl);
                        }
                        // --- Sub-block 1 ---
                        {
                            vuint8mf2_t v_q2 = __riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(v_rhs_তথ্য, 2, vl), 3, vl);
                            vint16m1_t v_w = __riscv_vmul_vv_i16m1(
                                __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(v_q2, vl)),
                                __riscv_vreinterpret_v_u16m1_i16m1(v_d_sb_1), vl);

                            ধ্রুবক int8_t* q8 = &lhs_qs_ptr[(base_k_phase + k_offsets[1] + l) * 4];
                            v_isum_0 = __riscv_vwmacc_vx_i32m2(v_isum_0, (int16_t)q8[0], v_w, vl);
                            v_isum_1 = __riscv_vwmacc_vx_i32m2(v_isum_1, (int16_t)q8[1], v_w, vl);
                            v_isum_2 = __riscv_vwmacc_vx_i32m2(v_isum_2, (int16_t)q8[2], v_w, vl);
                            v_isum_3 = __riscv_vwmacc_vx_i32m2(v_isum_3, (int16_t)q8[3], v_w, vl);
                        }
                        // --- Sub-block 2 ---
                        {
                            vuint8mf2_t v_q2 = __riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(v_rhs_তথ্য, 4, vl), 3, vl);
                            vint16m1_t v_w = __riscv_vmul_vv_i16m1(
                                __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(v_q2, vl)),
                                __riscv_vreinterpret_v_u16m1_i16m1(v_d_sb_2), vl);

                            ধ্রুবক int8_t* q8 = &lhs_qs_ptr[(base_k_phase + k_offsets[2] + l) * 4];
                            v_isum_0 = __riscv_vwmacc_vx_i32m2(v_isum_0, (int16_t)q8[0], v_w, vl);
                            v_isum_1 = __riscv_vwmacc_vx_i32m2(v_isum_1, (int16_t)q8[1], v_w, vl);
                            v_isum_2 = __riscv_vwmacc_vx_i32m2(v_isum_2, (int16_t)q8[2], v_w, vl);
                            v_isum_3 = __riscv_vwmacc_vx_i32m2(v_isum_3, (int16_t)q8[3], v_w, vl);
                        }
                        // --- Sub-block 3 ---
                        {
                            vuint8mf2_t v_q2 = __riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(v_rhs_তথ্য, 6, vl), 3, vl);
                            vint16m1_t v_w = __riscv_vmul_vv_i16m1(
                                __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(v_q2, vl)),
                                __riscv_vreinterpret_v_u16m1_i16m1(v_d_sb_3), vl);

                            ধ্রুবক int8_t* q8 = &lhs_qs_ptr[(base_k_phase + k_offsets[3] + l) * 4];
                            v_isum_0 = __riscv_vwmacc_vx_i32m2(v_isum_0, (int16_t)q8[0], v_w, vl);
                            v_isum_1 = __riscv_vwmacc_vx_i32m2(v_isum_1, (int16_t)q8[1], v_w, vl);
                            v_isum_2 = __riscv_vwmacc_vx_i32m2(v_isum_2, (int16_t)q8[2], v_w, vl);
                            v_isum_3 = __riscv_vwmacc_vx_i32m2(v_isum_3, (int16_t)q8[3], v_w, vl);
                        }
                    }

                    // C CORRECTION
                    পূর্ণসংখ্যা sb_base_abs = base_k_phase / 16;

                    // --- Correction Sub-block 0 ---
                    {
                        পূর্ণসংখ্যা sb_abs = sb_base_abs + (k_offsets[0] / 16);
                        vint16m1_t v_min = __riscv_vreinterpret_v_u16m1_i16m1(v_m_sb_0);

                        // Row 0
                        vfloat32m2_t v_g_min = __riscv_vfmul_vf_f32m2(v_g_min_base, lhs_current->d[0], vl);
                        vint32m2_t v_c = __riscv_vwmul_vx_i32m2(v_min, lhs_current->bsums[sb_abs * 4 + 0], vl);
                        vfloat32m2_t vf_c = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(v_c, vl), v_g_min, vl);
                        v_sumf_0 = __riscv_vfsub_vv_f32m2(v_sumf_0, vf_c, vl);

                        // Row 1
                        v_g_min = __riscv_vfmul_vf_f32m2(v_g_min_base, lhs_current->d[1], vl);
                        v_c = __riscv_vwmul_vx_i32m2(v_min, lhs_current->bsums[sb_abs * 4 + 1], vl);
                        vf_c = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(v_c, vl), v_g_min, vl);
                        v_sumf_1 = __riscv_vfsub_vv_f32m2(v_sumf_1, vf_c, vl);

                        // Row 2
                        v_g_min = __riscv_vfmul_vf_f32m2(v_g_min_base, lhs_current->d[2], vl);
                        v_c = __riscv_vwmul_vx_i32m2(v_min, lhs_current->bsums[sb_abs * 4 + 2], vl);
                        vf_c = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(v_c, vl), v_g_min, vl);
                        v_sumf_2 = __riscv_vfsub_vv_f32m2(v_sumf_2, vf_c, vl);

                        // Row 3
                        v_g_min = __riscv_vfmul_vf_f32m2(v_g_min_base, lhs_current->d[3], vl);
                        v_c = __riscv_vwmul_vx_i32m2(v_min, lhs_current->bsums[sb_abs * 4 + 3], vl);
                        vf_c = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(v_c, vl), v_g_min, vl);
                        v_sumf_3 = __riscv_vfsub_vv_f32m2(v_sumf_3, vf_c, vl);
                    }

                    // --- Correction Sub-block 1 ---
                    {
                        পূর্ণসংখ্যা sb_abs = sb_base_abs + (k_offsets[1] / 16);
                        vint16m1_t v_min = __riscv_vreinterpret_v_u16m1_i16m1(v_m_sb_1);

                        vfloat32m2_t v_g_min = __riscv_vfmul_vf_f32m2(v_g_min_base, lhs_current->d[0], vl);
                        vint32m2_t v_c = __riscv_vwmul_vx_i32m2(v_min, lhs_current->bsums[sb_abs * 4 + 0], vl);
                        vfloat32m2_t vf_c = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(v_c, vl), v_g_min, vl);
                        v_sumf_0 = __riscv_vfsub_vv_f32m2(v_sumf_0, vf_c, vl);

                        v_g_min = __riscv_vfmul_vf_f32m2(v_g_min_base, lhs_current->d[1], vl);
                        v_c = __riscv_vwmul_vx_i32m2(v_min, lhs_current->bsums[sb_abs * 4 + 1], vl);
                        vf_c = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(v_c, vl), v_g_min, vl);
                        v_sumf_1 = __riscv_vfsub_vv_f32m2(v_sumf_1, vf_c, vl);

                        v_g_min = __riscv_vfmul_vf_f32m2(v_g_min_base, lhs_current->d[2], vl);
                        v_c = __riscv_vwmul_vx_i32m2(v_min, lhs_current->bsums[sb_abs * 4 + 2], vl);
                        vf_c = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(v_c, vl), v_g_min, vl);
                        v_sumf_2 = __riscv_vfsub_vv_f32m2(v_sumf_2, vf_c, vl);

                        v_g_min = __riscv_vfmul_vf_f32m2(v_g_min_base, lhs_current->d[3], vl);
                        v_c = __riscv_vwmul_vx_i32m2(v_min, lhs_current->bsums[sb_abs * 4 + 3], vl);
                        vf_c = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(v_c, vl), v_g_min, vl);
                        v_sumf_3 = __riscv_vfsub_vv_f32m2(v_sumf_3, vf_c, vl);
                    }

                    // --- Correction Sub-block 2 ---
                    {
                        পূর্ণসংখ্যা sb_abs = sb_base_abs + (k_offsets[2] / 16);
                        vint16m1_t v_min = __riscv_vreinterpret_v_u16m1_i16m1(v_m_sb_2);

                        vfloat32m2_t v_g_min = __riscv_vfmul_vf_f32m2(v_g_min_base, lhs_current->d[0], vl);
                        vint32m2_t v_c = __riscv_vwmul_vx_i32m2(v_min, lhs_current->bsums[sb_abs * 4 + 0], vl);
                        vfloat32m2_t vf_c = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(v_c, vl), v_g_min, vl);
                        v_sumf_0 = __riscv_vfsub_vv_f32m2(v_sumf_0, vf_c, vl);

                        v_g_min = __riscv_vfmul_vf_f32m2(v_g_min_base, lhs_current->d[1], vl);
                        v_c = __riscv_vwmul_vx_i32m2(v_min, lhs_current->bsums[sb_abs * 4 + 1], vl);
                        vf_c = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(v_c, vl), v_g_min, vl);
                        v_sumf_1 = __riscv_vfsub_vv_f32m2(v_sumf_1, vf_c, vl);

                        v_g_min = __riscv_vfmul_vf_f32m2(v_g_min_base, lhs_current->d[2], vl);
                        v_c = __riscv_vwmul_vx_i32m2(v_min, lhs_current->bsums[sb_abs * 4 + 2], vl);
                        vf_c = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(v_c, vl), v_g_min, vl);
                        v_sumf_2 = __riscv_vfsub_vv_f32m2(v_sumf_2, vf_c, vl);

                        v_g_min = __riscv_vfmul_vf_f32m2(v_g_min_base, lhs_current->d[3], vl);
                        v_c = __riscv_vwmul_vx_i32m2(v_min, lhs_current->bsums[sb_abs * 4 + 3], vl);
                        vf_c = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(v_c, vl), v_g_min, vl);
                        v_sumf_3 = __riscv_vfsub_vv_f32m2(v_sumf_3, vf_c, vl);
                    }

                    // --- Correction Sub-block 3 ---
                    {
                        পূর্ণসংখ্যা sb_abs = sb_base_abs + (k_offsets[3] / 16);
                        vint16m1_t v_min = __riscv_vreinterpret_v_u16m1_i16m1(v_m_sb_3);

                        vfloat32m2_t v_g_min = __riscv_vfmul_vf_f32m2(v_g_min_base, lhs_current->d[0], vl);
                        vint32m2_t v_c = __riscv_vwmul_vx_i32m2(v_min, lhs_current->bsums[sb_abs * 4 + 0], vl);
                        vfloat32m2_t vf_c = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(v_c, vl), v_g_min, vl);
                        v_sumf_0 = __riscv_vfsub_vv_f32m2(v_sumf_0, vf_c, vl);

                        v_g_min = __riscv_vfmul_vf_f32m2(v_g_min_base, lhs_current->d[1], vl);
                        v_c = __riscv_vwmul_vx_i32m2(v_min, lhs_current->bsums[sb_abs * 4 + 1], vl);
                        vf_c = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(v_c, vl), v_g_min, vl);
                        v_sumf_1 = __riscv_vfsub_vv_f32m2(v_sumf_1, vf_c, vl);

                        v_g_min = __riscv_vfmul_vf_f32m2(v_g_min_base, lhs_current->d[2], vl);
                        v_c = __riscv_vwmul_vx_i32m2(v_min, lhs_current->bsums[sb_abs * 4 + 2], vl);
                        vf_c = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(v_c, vl), v_g_min, vl);
                        v_sumf_2 = __riscv_vfsub_vv_f32m2(v_sumf_2, vf_c, vl);

                        v_g_min = __riscv_vfmul_vf_f32m2(v_g_min_base, lhs_current->d[3], vl);
                        v_c = __riscv_vwmul_vx_i32m2(v_min, lhs_current->bsums[sb_abs * 4 + 3], vl);
                        vf_c = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(v_c, vl), v_g_min, vl);
                        v_sumf_3 = __riscv_vfsub_vv_f32m2(v_sumf_3, vf_c, vl);
                    }

                } // End Phase Loop

                // --- Apply Main Scales ---
                vfloat16m1_t v_g_all_f16 = __riscv_vle16_v_f16m1((ধ্রুবক _Float16*)rhs_current->d, vl);
                vfloat32m2_t v_g_all_base = __riscv_vfwcvt_f_f_v_f32m2(v_g_all_f16, vl);

                {
                    vfloat32m2_t v_g_all = __riscv_vfmul_vf_f32m2(v_g_all_base, lhs_current->d[0], vl);
                    vfloat32m2_t v_যোগ = __riscv_vfcvt_f_x_v_f32m2(v_isum_0, vl);
                    v_যোগ = __riscv_vfmul_vv_f32m2(v_যোগ, v_g_all, vl);
                    v_sumf_0 = __riscv_vfadd_vv_f32m2(v_sumf_0, v_যোগ, vl);
                }
                // Row 1
                {
                    vfloat32m2_t v_g_all = __riscv_vfmul_vf_f32m2(v_g_all_base, lhs_current->d[1], vl);
                    vfloat32m2_t v_যোগ = __riscv_vfcvt_f_x_v_f32m2(v_isum_1, vl);
                    v_যোগ = __riscv_vfmul_vv_f32m2(v_যোগ, v_g_all, vl);
                    v_sumf_1 = __riscv_vfadd_vv_f32m2(v_sumf_1, v_যোগ, vl);
                }
                // Row 2
                {
                    vfloat32m2_t v_g_all = __riscv_vfmul_vf_f32m2(v_g_all_base, lhs_current->d[2], vl);
                    vfloat32m2_t v_যোগ = __riscv_vfcvt_f_x_v_f32m2(v_isum_2, vl);
                    v_যোগ = __riscv_vfmul_vv_f32m2(v_যোগ, v_g_all, vl);
                    v_sumf_2 = __riscv_vfadd_vv_f32m2(v_sumf_2, v_যোগ, vl);
                }
                // Row 3
                {
                    vfloat32m2_t v_g_all = __riscv_vfmul_vf_f32m2(v_g_all_base, lhs_current->d[3], vl);
                    vfloat32m2_t v_যোগ = __riscv_vfcvt_f_x_v_f32m2(v_isum_3, vl);
                    v_যোগ = __riscv_vfmul_vv_f32m2(v_যোগ, v_g_all, vl);
                    v_sumf_3 = __riscv_vfadd_vv_f32m2(v_sumf_3, v_যোগ, vl);
                }

            } // End K-Block

            __riscv_vse32_v_f32m2(s + (row_tile + 0) * bs + col_tile, v_sumf_0, vl);
            __riscv_vse32_v_f32m2(s + (row_tile + 1) * bs + col_tile, v_sumf_1, vl);
            __riscv_vse32_v_f32m2(s + (row_tile + 2) * bs + col_tile, v_sumf_2, vl);
            __riscv_vse32_v_f32m2(s + (row_tile + 3) * bs + col_tile, v_sumf_3, vl);
        }
    }
}
#endif
