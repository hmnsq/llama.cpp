#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_IMPL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "ggml-quants.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-cpu.h"
#ಅನ್ತರ್ಭಾವ "simd-mappings.h"

#ಅನ್ತರ್ಭಾವ "../../quants.h"
#ಅನ್ತರ್ಭಾವ "../../ggml-cpu-impl.h"

#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <string.h>
#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <float.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h> // for qsort
#ಅನ್ತರ್ಭಾವ <stdio.h>  // for GGML_ASSERT

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _MSC_VER
#ವ್ಯಾಖ್ಯಾನಿಸು NOINLINE __declspec(noinline)
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು NOINLINE __attribute__((__noinline__))
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿಸು ಗುಂಪು_MAX_EPS 1e-15f
#ವ್ಯಾಖ್ಯಾನಿಸು ಗುಂಪು_MAX_EPS_IQ3_XXS 1e-8f
#ವ್ಯಾಖ್ಯಾನಿಸು ಗುಂಪು_MAX_EPS_IQ2_S 1e-8f
#ವ್ಯಾಖ್ಯಾನಿಸು ಗುಂಪು_MAX_EPS_IQ1_M 1e-7f
#ವ್ಯಾಖ್ಯಾನಿಸು ಗುಂಪು_MAX_EPS_IQ1_S 1e-12f

#ವ್ಯಾಖ್ಯಾನಿಸು ಬಳಸದ GGML_ಬಳಸದ

ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q8_0(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT vy, int64_t k) {
    ಪ್ರತಿಜ್ಞಾ(QK8_0 == 32);
    ಪ್ರತಿಜ್ಞಾ(k % QK8_0 == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = k / QK8_0;

    ಖಂಡ_q8_0 * GGML_RESTRICT y = vy;

#ಆದರೆ defined(__riscv_v)

    size_t vl = QK8_0;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; i++) {
        // load elements
        vfloat32m8_t v_x   = __riscv_vle32_v_f32m8(x+i*QK8_0, vl);

        vfloat32m8_t vfabs = __riscv_vfabs_v_f32m8(v_x, vl);
        vfloat32m1_t tmp   = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t vmax  = __riscv_vfredmax_vs_f32m8_f32m1(vfabs, tmp, vl);
        ದಶಮಾಂಶ amax = __riscv_vfmv_f_s_f32m1_f32(vmax);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = amax / ((1 << 7) - 1);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_CPU_FP32_ಗಮ್ಯ_FP16(d);

        vfloat32m8_t x0 = __riscv_vfmul_vf_f32m8(v_x, id, vl);

        // convert to integer
        vint16m4_t   vi = __riscv_vfncvt_x_f_w_i16m4(x0, vl);
        vint8m2_t    vs = __riscv_vncvt_x_x_w_i8m2(vi, vl);

        // store result
        __riscv_vse8_v_i8m2(y[i].qs , vs, vl);
    }
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ಬಳಸದ(nb);
    // scalar
    quantize_ಅಡ್ಡಸಾಲು_q8_0_ref(x, y, k);
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q8_1(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT vy, int64_t k) {
    ಪ್ರತಿಜ್ಞಾ(k % QK8_1 == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = k / QK8_1;

    ಖಂಡ_q8_1 * GGML_RESTRICT y = vy;

#ಆದರೆ defined(__riscv_v)

    size_t vl = QK8_1;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; i++) {
        // load elements
        vfloat32m8_t v_x   = __riscv_vle32_v_f32m8(x+i*QK8_1, vl);

        vfloat32m8_t vfabs = __riscv_vfabs_v_f32m8(v_x, vl);
        vfloat32m1_t tmp   = __riscv_vfmv_v_f_f32m1(0.0, vl);
        vfloat32m1_t vmax  = __riscv_vfredmax_vs_f32m8_f32m1(vfabs, tmp, vl);
        ದಶಮಾಂಶ amax = __riscv_vfmv_f_s_f32m1_f32(vmax);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d  = amax / ((1 << 7) - 1);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_CPU_FP32_ಗಮ್ಯ_FP16(d);

        vfloat32m8_t x0 = __riscv_vfmul_vf_f32m8(v_x, id, vl);

        // convert to integer
        vint16m4_t   vi = __riscv_vfncvt_x_f_w_i16m4(x0, vl);
        vint8m2_t    vs = __riscv_vncvt_x_x_w_i8m2(vi, vl);

        // store result
        __riscv_vse8_v_i8m2(y[i].qs , vs, vl);

        // compute sum for y[i].s
        vint16m1_t tmp2 = __riscv_vmv_v_x_i16m1(0, vl);
        vint16m1_t vwrs = __riscv_vwredsum_vs_i8m2_i16m1(vs, tmp2, vl);

        // set y[i].s
        ಪೂರ್ಣಾಂಕ ಮೊತ್ತ = __riscv_vmv_x_s_i16m1_i16(vwrs);
        y[i].s = GGML_CPU_FP32_ಗಮ್ಯ_FP16(ಮೊತ್ತ*d);
    }

#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ಬಳಸದ(nb);
    // scalar
    quantize_ಅಡ್ಡಸಾಲು_q8_1_ref(x, y, k);
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q8_K(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k) {
    ಪ್ರತಿಜ್ಞಾ(k % QK_K == 0);
    size_t nb = k / QK_K;

#ಆದರೆ defined __riscv_v
    ಖಂಡ_q8_K * y_ಖಂಡಗಳು = (ಖಂಡ_q8_K *)y;
    ಸ್ಥಿರಾಂಕ size_t vlmax_f32m8 = __riscv_vsetvlmax_e32m8();

    ಗಾಗಿ (size_t i = 0; i < nb; i++) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* x_ಖಂಡ = x + i * QK_K;
        ಖಂಡ_q8_K* y_ಖಂಡ = &y_ಖಂಡಗಳು[i];

        // 1. Calculate Min/Max
        vfloat32m8_t max_v = __riscv_vfmv_v_f_f32m8(-__builtin_inff(), vlmax_f32m8);
        vfloat32m8_t min_v = __riscv_vfmv_v_f_f32m8(__builtin_inff(), vlmax_f32m8);

        size_t rem = QK_K;
        size_t ಸ್ಥಾನಾಂತರ = 0;
        ಆಗ (rem > 0) {
            size_t vl = __riscv_vsetvl_e32m8(rem);
            vfloat32m8_t v_curr = __riscv_vle32_v_f32m8(x_ಖಂಡ + ಸ್ಥಾನಾಂತರ, vl);
            max_v = __riscv_vfmax_vv_f32m8(max_v, v_curr, vl);
            min_v = __riscv_vfmin_vv_f32m8(min_v, v_curr, vl);
            rem -= vl;
            ಸ್ಥಾನಾಂತರ += vl;
        }

        vfloat32m1_t v_init_max = __riscv_vfmv_s_f_f32m1(-__builtin_inff(), 1);
        vfloat32m1_t v_init_min = __riscv_vfmv_s_f_f32m1(__builtin_inff(), 1);

        vfloat32m1_t v_scalar_max = __riscv_vfredmax_vs_f32m8_f32m1(max_v, v_init_max, vlmax_f32m8);
        vfloat32m1_t v_scalar_min = __riscv_vfredmin_vs_f32m8_f32m1(min_v, v_init_min, vlmax_f32m8);

        ದಶಮಾಂಶ max_val = __riscv_vfmv_f_s_f32m1_f32(v_scalar_max);
        ದಶಮಾಂಶ min_val = __riscv_vfmv_f_s_f32m1_f32(v_scalar_min);

        ದಶಮಾಂಶ amax = fabsf(max_val) > fabsf(min_val) ? fabsf(max_val) : fabsf(min_val);

        ಆದರೆ (amax == 0.0f) {
            y_ಖಂಡ->d = 0.0f;
            memset(y_ಖಂಡ->qs, 0, QK_K);
            memset(y_ಖಂಡ->bsums, 0, ಗಾತ್ರ(y_ಖಂಡ->bsums));
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ iscale = -127.f / (fabsf(max_val) > fabsf(min_val) ? max_val : min_val);
        y_ಖಂಡ->d = 1.0f / iscale;

        // 2. Quantize and Calculate Sums
        ಸ್ಥಾನಾಂತರ = 0;
        rem = QK_K;
        vint16m1_t v_zero_ಮೊತ್ತ = __riscv_vmv_v_x_i16m1(0, 1);

        ಆಗ (rem > 0) {
            size_t vl = __riscv_vsetvl_e32m8(rem);
            vfloat32m8_t v_f = __riscv_vle32_v_f32m8(x_ಖಂಡ + ಸ್ಥಾನಾಂತರ, vl);

            v_f = __riscv_vfmul_vf_f32m8(v_f, iscale, vl);

            vint32m8_t v_i32 = __riscv_vfcvt_x_f_v_i32m8_rm(v_f, __RISCV_FRM_RNE, vl);
            vint16m4_t v_i16 = __riscv_vnclip_wx_i16m4(v_i32, 0, __RISCV_VXRM_RNE, vl);
            vint8m2_t v_q = __riscv_vnclip_wx_i8m2(v_i16, 0, __RISCV_VXRM_RNE, vl);

            __riscv_vse8_v_i8m2(y_ಖಂಡ->qs + ಸ್ಥಾನಾಂತರ, v_q, vl);

            // first iteration clear

            ಪೂರ್ಣಾಂಕ ಮೊತ್ತ_idx;
            vint8m1_t chunk_m1;
            vint16m1_t v_ಮೊತ್ತ;
            ಮೊತ್ತ_idx = ಸ್ಥಾನಾಂತರ / 16;
            chunk_m1 = __riscv_vget_v_i8m2_i8m1(v_q, 0);
            v_ಮೊತ್ತ = __riscv_vwredsum_vs_i8m1_i16m1(chunk_m1, v_zero_ಮೊತ್ತ, 16);
            y_ಖಂಡ->bsums[ಮೊತ್ತ_idx] = (int16_t)__riscv_vmv_x_s_i16m1_i16(v_ಮೊತ್ತ);

            // remaining iterations
            vint8m2_t slid_q = v_q;
            ಗಾಗಿ (size_t k = 16; k < vl; k += 16) {
                slid_q = __riscv_vslidedown_vx_i8m2(slid_q, 16, vl);

                ಮೊತ್ತ_idx = (ಸ್ಥಾನಾಂತರ + k) / 16;
                chunk_m1 = __riscv_vget_v_i8m2_i8m1(slid_q, 0);

                v_ಮೊತ್ತ = __riscv_vwredsum_vs_i8m1_i16m1(chunk_m1, v_zero_ಮೊತ್ತ, 16);
                y_ಖಂಡ->bsums[ಮೊತ್ತ_idx] =(int16_t)__riscv_vmv_x_s_i16m1_i16(v_ಮೊತ್ತ);
            }

            rem -= vl;
            ಸ್ಥಾನಾಂತರ += vl;
        }
    }
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ಬಳಸದ(nb);
    // scalar
    quantize_ಅಡ್ಡಸಾಲು_q8_K_ref(x, y, k);
#ಆದರೆ_ಅಂತ್ಯ
}

//===================================== Dot products =================================

ಶೂನ್ಯ ggml_vec_dot_q4_0_q8_0(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
#ಆದರೆ defined(__riscv_v)
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qk = QK8_0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / qk;

    ಪ್ರತಿಜ್ಞಾ(n % qk == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * GGML_RESTRICT y = vy;

    ಪೂರ್ಣಾಂಕ ib = 0;
    ದಶಮಾಂಶ sumf = 0;

    size_t vl = qk / 2;

    ಗಾಗಿ (; ib < nb; ++ib) {
        // load elements
        vuint8m1_t tx = __riscv_vle8_v_u8m1(x[ib].qs, vl);

        vint8m1_t y0 = __riscv_vle8_v_i8m1(y[ib].qs, vl);
        vint8m1_t y1 = __riscv_vle8_v_i8m1(y[ib].qs+16, vl);

        // mask and store lower part of x, and then upper part
        vuint8m1_t x_a = __riscv_vand_vx_u8m1(tx, 0x0F, vl);
        vuint8m1_t x_l = __riscv_vsrl_vx_u8m1(tx, 0x04, vl);

        vint8m1_t x_ai = __riscv_vreinterpret_v_u8m1_i8m1(x_a);
        vint8m1_t x_li = __riscv_vreinterpret_v_u8m1_i8m1(x_l);

        // subtract offset
        vint8m1_t v0 = __riscv_vsub_vx_i8m1(x_ai, 8, vl);
        vint8m1_t v1 = __riscv_vsub_vx_i8m1(x_li, 8, vl);

        vint16m2_t vec_mul1 = __riscv_vwmul_vv_i16m2(v0, y0, vl);
        vint16m2_t vec_mul2 = __riscv_vwmacc_vv_i16m2(vec_mul1, v1, y1, vl);

        vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t vs2 = __riscv_vwredsum_vs_i16m2_i32m1(vec_mul2, vec_zero, vl);

        ಪೂರ್ಣಾಂಕ sumi = __riscv_vmv_x_s_i32m1_i32(vs2);

        sumf += sumi*GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[ib].d)*GGML_CPU_FP16_ಗಮ್ಯ_FP32(y[ib].d);
    }

    *s = sumf;
#ಇಲ್ಲದಿದ್ದರೆ
    ggml_vec_dot_q4_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_vec_dot_q4_1_q8_1(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
#ಆದರೆ defined(__riscv_v)
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qk = QK8_1;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / qk;

    ಪ್ರತಿಜ್ಞಾ(n % qk == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * GGML_RESTRICT y = vy;

    ಪೂರ್ಣಾಂಕ ib = 0;
    ದಶಮಾಂಶ sumf = 0;

    size_t vl = qk / 2;

    ಗಾಗಿ (; ib < nb; ++ib) {
        // load elements
        vuint8m1_t tx = __riscv_vle8_v_u8m1(x[ib].qs, vl);

        vint8m1_t y0 = __riscv_vle8_v_i8m1(y[ib].qs, vl);
        vint8m1_t y1 = __riscv_vle8_v_i8m1(y[ib].qs+16, vl);

        // mask and store lower part of x, and then upper part
        vuint8m1_t x_a = __riscv_vand_vx_u8m1(tx, 0x0F, vl);
        vuint8m1_t x_l = __riscv_vsrl_vx_u8m1(tx, 0x04, vl);

        vint8m1_t v0 = __riscv_vreinterpret_v_u8m1_i8m1(x_a);
        vint8m1_t v1 = __riscv_vreinterpret_v_u8m1_i8m1(x_l);

        vint16m2_t vec_mul1 = __riscv_vwmul_vv_i16m2(v0, y0, vl);
        vint16m2_t vec_mul2 = __riscv_vwmacc_vv_i16m2(vec_mul1, v1, y1, vl);

        vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t vs2 = __riscv_vwredsum_vs_i16m2_i32m1(vec_mul2, vec_zero, vl);

        ಪೂರ್ಣಾಂಕ sumi = __riscv_vmv_x_s_i32m1_i32(vs2);

        sumf += (GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[ib].d)*GGML_CPU_FP16_ಗಮ್ಯ_FP32(y[ib].d))*sumi + GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[ib].m)*GGML_CPU_FP16_ಗಮ್ಯ_FP32(y[ib].s);
    }

    *s = sumf;
#ಇಲ್ಲದಿದ್ದರೆ
    ggml_vec_dot_q4_1_q8_1_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_vec_dot_q5_0_q8_0(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
#ಆದರೆ defined(__riscv_v)
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qk = QK8_0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / qk;

    ಪೂರ್ಣಾಂಕ ib = 0;
    ದಶಮಾಂಶ sumf = 0;

    ಪ್ರತಿಜ್ಞಾ(n % qk == 0);
    ಪ್ರತಿಜ್ಞಾ(qk == QK5_0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_0 * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * GGML_RESTRICT y = vy;

    size_t vl;
    size_t vlenb = __riscv_vlenb();

    ಗಾಗಿ (; ib < nb; ++ib) {
        vl = qk / 2;
        vuint8m1_t v0 = __riscv_vle8_v_u8m1(x[ib].qs, vl);
        vint8m1_t v0l = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(v0, 0x0F, vl));
        vint8m1_t v0h = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(v0, 4, vl));
        vint8m2_t v0c;
        ಆದರೆ (vlenb == 16) {
            v0c = __riscv_vcreate_v_i8m1_i8m2(v0l, v0h);
        } ಇಲ್ಲದಿದ್ದರೆ {
            v0l = __riscv_vslideup_vx_i8m1(v0l, v0h, 16, 32);
            v0c = __riscv_vlmul_ext_v_i8m1_i8m2(v0l);
        }

        vl = qk;
        vbool4_t qh = __riscv_vlm_v_b4(x[ib].qh, vl);
        qh = __riscv_vmnand_mm_b4(qh, qh, vl);
        vint8m2_t v0f = __riscv_vsub_vx_i8m2_mu(qh, v0c, v0c, 0x10, vl);
        vint8m2_t v1 = __riscv_vle8_v_i8m2(y[ib].qs, vl);
        vint16m4_t mul = __riscv_vwmul_vv_i16m4(v0f, v1, vl);
        vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t ಮೊತ್ತ = __riscv_vwredsum_vs_i16m4_i32m1(mul, zero, vl);
        int32_t sumi = __riscv_vmv_x_s_i32m1_i32(ಮೊತ್ತ);

        sumf += (GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[ib].d) * GGML_CPU_FP16_ಗಮ್ಯ_FP32(y[ib].d)) * sumi;
    }

    *s = sumf;
#ಇಲ್ಲದಿದ್ದರೆ
    ggml_vec_dot_q5_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_vec_dot_q5_1_q8_1(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
#ಆದರೆ defined(__riscv_v)
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qk = QK8_1;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / qk;

    ಪೂರ್ಣಾಂಕ ib = 0;
    ದಶಮಾಂಶ sumf = 0;

    ಪ್ರತಿಜ್ಞಾ(n % qk == 0);
    ಪ್ರತಿಜ್ಞಾ(qk == QK5_1);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_1 * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * GGML_RESTRICT y = vy;

    size_t vl;
    size_t vlenb = __riscv_vlenb();

    ಗಾಗಿ (; ib < nb; ++ib) {
        vl = qk / 2;
        vuint8m1_t v0 = __riscv_vle8_v_u8m1(x[ib].qs, vl);
        vint8m1_t v0l = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(v0, 0x0F, vl));
        vint8m1_t v0h = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(v0, 4, vl));
        vint8m2_t v0c;
        ಆದರೆ (vlenb == 16) {
            v0c = __riscv_vcreate_v_i8m1_i8m2(v0l, v0h);
        } ಇಲ್ಲದಿದ್ದರೆ {
            v0l = __riscv_vslideup_vx_i8m1(v0l, v0h, 16, 32);
            v0c = __riscv_vlmul_ext_v_i8m1_i8m2(v0l);
        }

        vl = qk;
        vbool4_t qh = __riscv_vlm_v_b4(x[ib].qh, vl);
        vint8m2_t v0f = __riscv_vor_vx_i8m2_mu(qh, v0c, v0c, 0x10, vl);
        vint8m2_t v1 = __riscv_vle8_v_i8m2(y[ib].qs, vl);
        vint16m4_t mul = __riscv_vwmul_vv_i16m4(v0f, v1, vl);
        vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t ಮೊತ್ತ = __riscv_vwredsum_vs_i16m4_i32m1(mul, zero, vl);
        int32_t sumi = __riscv_vmv_x_s_i32m1_i32(ಮೊತ್ತ);

        sumf += (GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[ib].d)*GGML_CPU_FP16_ಗಮ್ಯ_FP32(y[ib].d))*sumi + GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[ib].m)*GGML_CPU_FP16_ಗಮ್ಯ_FP32(y[ib].s);
    }

    *s = sumf;
#ಇಲ್ಲದಿದ್ದರೆ
    ggml_vec_dot_q5_1_q8_1_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_vec_dot_q8_0_q8_0(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qk = QK8_0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / qk;

    ಪ್ರತಿಜ್ಞಾ(n % qk == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * GGML_RESTRICT y = vy;

    ಪೂರ್ಣಾಂಕ ib = 0;
    ದಶಮಾಂಶ sumf = 0;

#ಆದರೆ defined(__riscv_v)
    size_t vl = qk;

    ಗಾಗಿ (; ib < nb; ++ib) {
        // load elements
        vint8m2_t bx_0 = __riscv_vle8_v_i8m2(x[ib].qs, vl);
        vint8m2_t by_0 = __riscv_vle8_v_i8m2(y[ib].qs, vl);

        vint16m4_t vw_mul = __riscv_vwmul_vv_i16m4(bx_0, by_0, vl);

        vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t v_ಮೊತ್ತ = __riscv_vwredsum_vs_i16m4_i32m1(vw_mul, v_zero, vl);

        ಪೂರ್ಣಾಂಕ sumi = __riscv_vmv_x_s_i32m1_i32(v_ಮೊತ್ತ);

        sumf += sumi*(GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[ib].d)*GGML_CPU_FP16_ಗಮ್ಯ_FP32(y[ib].d));
    }

    *s = sumf;
#ಇಲ್ಲದಿದ್ದರೆ

    ಬಳಸದ(nb);
    ಬಳಸದ(x);
    ಬಳಸದ(y);
    ಬಳಸದ(ib);
    ಬಳಸದ(sumf);

    ggml_vec_dot_q8_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

#ಆದರೆ defined(__riscv_v)
ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_q1_0_q8_0_vl256(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qk = QK1_0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / qk;
    ಪ್ರತಿಜ್ಞಾ(n % qk == 0);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q1_0 * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * GGML_RESTRICT y = vy;

    //LMUL = 1, VLMAX = 32
    ಸ್ಥಿರಾಂಕ size_t vl32 = __riscv_vsetvl_e8m1(32);
    ಪ್ರತಿಜ್ಞಾ(vl32 == 32);

    ಸ್ಥಿರಾಂಕ vint16m1_t zero = __riscv_vmv_v_x_i16m1(0, 1);

    ದಶಮಾಂಶ sumf = 0;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ib = 0; ib < nb; ++ib) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d0 = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[ib].d);

        ದಶಮಾಂಶ acc = 0;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 4; ++k) {
            ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * GGML_RESTRICT yb = &y[ib * 4 + k];
            ಸ್ಥಿರಾಂಕ vbool8_t is_not_zero = __riscv_vlm_v_b8(x[ib].qs + 4 * k, vl32);

            ಸ್ಥಿರಾಂಕ vint8m1_t qy = __riscv_vle8_v_i8m1(yb->qs, vl32);
            ಸ್ಥಿರಾಂಕ vint8m1_t neg_qy = __riscv_vneg_v_i8m1(qy, vl32);
            ಸ್ಥಿರಾಂಕ vint8m1_t sy = __riscv_vmerge_vvm_i8m1(neg_qy, qy, is_not_zero, vl32);

            ಸ್ಥಿರಾಂಕ vint16m1_t red = __riscv_vwredsum_vs_i8m1_i16m1(sy, zero, vl32);
            acc += GGML_CPU_FP16_ಗಮ್ಯ_FP32(yb->d) * (ದಶಮಾಂಶ)__riscv_vmv_x_s_i16m1_i16(red);
        }

        sumf += d0 * acc;
    }

    *s = sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_q1_0_q8_0_vl128(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qk = QK1_0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / qk;
    ಪ್ರತಿಜ್ಞಾ(n % qk == 0);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q1_0 * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * GGML_RESTRICT y = vy;

    //LMUL = 2, VLMAX = 32
    ಸ್ಥಿರಾಂಕ size_t vl32 = __riscv_vsetvl_e8m2(32);
    ಪ್ರತಿಜ್ಞಾ(vl32 == 32);

    ಸ್ಥಿರಾಂಕ vint16m1_t zero = __riscv_vmv_v_x_i16m1(0, 1);

    ದಶಮಾಂಶ sumf = 0;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ib = 0; ib < nb; ++ib) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d0 = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[ib].d);

        ದಶಮಾಂಶ acc = 0;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 4; ++k) {
            ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * GGML_RESTRICT yb = &y[ib * 4 + k];
            ಸ್ಥಿರಾಂಕ vbool4_t is_not_zero = __riscv_vlm_v_b4(x[ib].qs + 4 * k, vl32);

            ಸ್ಥಿರಾಂಕ vint8m2_t qy = __riscv_vle8_v_i8m2(yb->qs, vl32);
            ಸ್ಥಿರಾಂಕ vint8m2_t neg_qy =__riscv_vneg_v_i8m2(qy, vl32);
            ಸ್ಥಿರಾಂಕ vint8m2_t sy = __riscv_vmerge_vvm_i8m2(neg_qy, qy, is_not_zero, vl32);

            ಸ್ಥಿರಾಂಕ vint16m1_t red = __riscv_vwredsum_vs_i8m2_i16m1(sy, zero, vl32);
            acc += GGML_CPU_FP16_ಗಮ್ಯ_FP32(yb->d) * (ದಶಮಾಂಶ)__riscv_vmv_x_s_i16m1_i16(red);
        }

        sumf += d0 * acc;
    }

    *s = sumf;
}
#ಆದರೆ_ಅಂತ್ಯ

ಶೂನ್ಯ ggml_vec_dot_q1_0_q8_0(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
#ಆದರೆ defined(__riscv_v)
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);

    ಸ್ಥಿರಾಂಕ size_t vlen_ಬಿಟ್ಟುಗಳು = __riscv_vlenb() * 8;

    ಆದರೆ (vlen_ಬಿಟ್ಟುಗಳು >= 256) {
        ggml_vec_dot_q1_0_q8_0_vl256(n, s, vx, vy);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (vlen_ಬಿಟ್ಟುಗಳು >= 128) {
        ggml_vec_dot_q1_0_q8_0_vl128(n, s, vx, vy);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ggml_vec_dot_q1_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ggml_vec_dot_q1_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

#ಆದರೆ defined __riscv_xtheadvector
ಶೂನ್ಯ ggml_vec_dot_q2_K_q8_K_xtheadvector(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_K * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    ದಶಮಾಂಶ sumf = 0;
    uint8_t atmp[16];

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ uint8_t * q2 = x[i].qs;
        ಸ್ಥಿರಾಂಕ  int8_t * q8 = y[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t * sc = x[i].scales;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dall = y[i].d * GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin = -y[i].d * GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].dmin);
        uint8_t *patmp = atmp;
        ಪೂರ್ಣಾಂಕ vsums;
        ಪೂರ್ಣಾಂಕ tmp;
        __asm__ __volatile__(
            "th.vsetvli zero, %[vl16], e8, m1\n\t"
            "th.vmv.v.x v8, zero\n\t"
            "th.vlb.v v1, (%[sc])\n\t"
            "th.vand.vi v0, v1, 0xF\n\t"
            "th.vsrl.vi v1, v1, 4\n\t"
            "th.vsb.v v0, (%[scale])\n\t"
            "th.vwaddu.vx v16, v1, zero\n\t"
            "th.vsetvli zero, %[vl16], e16, m2\n\t"
            "th.vlh.v v2, (%[bsums])\n\t"
            "th.vwmul.vv v4, v16, v2\n\t"
            "th.vsetvli zero, %[vl16], e32, m4\n\t"
            "th.vredsum.vs v8, v4, v8\n\t"
            "th.vmv.x.s %[vsums], v8"
            : [tmp] "=&r" (tmp), [vsums] "=&r" (vsums)
            : [sc] "r" (sc), [scale] "r" (atmp), [bsums] "r" (y[i].bsums)
            , [vl16] "r" (16)
            : "memory"
            , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
            , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
            , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
            , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
        );
        sumf += dmin * vsums;
        ಪೂರ್ಣಾಂಕ isum = 0;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K/128; ++j) {
            __asm__ __volatile__(
                "th.vsetvli zero, %[vl32], e8, m2\n\t"
                "th.vlb.v v0, (%[q2])\n\t"
                "th.vsrl.vi v2, v0, 2\n\t"
                "th.vsrl.vi v4, v0, 4\n\t"
                "th.vsrl.vi v6, v0, 6\n\t"
                "th.vand.vi v0, v0, 0x3\n\t"
                "th.vand.vi v2, v2, 0x3\n\t"
                "th.vand.vi v4, v4, 0x3\n\t"
                "th.vsetvli zero, %[vl128], e8, m8\n\t"
                "th.vlb.v v8, (%[q8])\n\t"
                "th.vsetvli zero, %[vl64], e8, m4\n\t"
                "th.vwmul.vv v16, v0, v8\n\t"
                "th.vwmul.vv v24, v4, v12\n\t"
                "th.vsetvli zero, %[vl16], e16, m2\n\t"
                "th.vmv.v.x v0, zero\n\t"
                "th.vwredsum.vs v10, v16, v0\n\t"
                "th.vwredsum.vs v9, v18, v0\n\t"
                "th.vwredsum.vs v8, v20, v0\n\t"
                "th.vwredsum.vs v7, v22, v0\n\t"
                "th.vwredsum.vs v11, v24, v0\n\t"
                "th.vwredsum.vs v12, v26, v0\n\t"
                "th.vwredsum.vs v13, v28, v0\n\t"
                "th.vwredsum.vs v14, v30, v0\n\t"
                "li %[tmp], 4\n\t"
                "th.vsetvli zero, %[tmp], e32, m1\n\t"
                "th.vslideup.vi v10, v9, 1\n\t"
                "th.vslideup.vi v8, v7, 1\n\t"
                "th.vslideup.vi v11, v12, 1\n\t"
                "th.vslideup.vi v13, v14, 1\n\t"
                "th.vslideup.vi v10, v8, 2\n\t"
                "th.vslideup.vi v11, v13, 2\n\t"
                "li %[tmp], 8\n\t"
                "th.vsetvli zero, %[tmp], e32, m2\n\t"
                "th.vlbu.v v12, (%[scale])\n\t"
                "th.vmul.vv v10, v10, v12\n\t"
                "th.vredsum.vs v0, v10, v0\n\t"
                "th.vmv.x.s %[tmp], v0\n\t"
                "add %[isum], %[isum], %[tmp]"
                : [tmp] "=&r" (tmp), [isum] "+&r" (isum)
                : [q2] "r" (q2), [scale] "r" (patmp), [q8] "r" (q8)
                , [vl16] "r" (16), [vl32] "r" (32), [vl64] "r" (64), [vl128] "r" (128)
                : "memory"
                , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );
            q2 += 32; q8 += 128; patmp += 8;
        }

        sumf += dall * isum;
    }

    *s = sumf;
}
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined __riscv_v
ಶೂನ್ಯ ggml_vec_dot_q2_K_q8_K_vl128(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_K * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    ದಶಮಾಂಶ sumf = 0;
    uint8_t atmp[16];

    uint8_t ತಾತ್ಕಾಲಿಕ_01[32] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ uint8_t * q2 = x[i].qs;
        ಸ್ಥಿರಾಂಕ  int8_t * q8 = y[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t * sc = x[i].scales;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dall = y[i].d * GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin = -y[i].d * GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].dmin);
        uint8_t *patmp = atmp;
        ಪೂರ್ಣಾಂಕ vsums;
        ಪೂರ್ಣಾಂಕ tmp, t1, t2, t3, t4, t5, t6, t7;
        __asm__ __volatile__(
            "vsetivli zero, 16, e8, m1\n\t"
            "vmv.v.x v8, zero\n\t"
            "lb zero, 15(%[sc])\n\t"
            "vle8.v v1, (%[sc])\n\t"
            "vle8.v v2, (%[bsums])\n\t"
            "addi %[tmp], %[bsums], 16\n\t"
            "vand.vi v0, v1, 0xF\n\t"
            "vsrl.vi v1, v1, 4\n\t"
            "vle8.v v3, (%[tmp])\n\t"
            "vse8.v v0, (%[scale])\n\t"
            "vsetivli zero, 16, e16, m2\n\t"
            "vzext.vf2 v0, v1\n\t"
            "vwmul.vv v4, v0, v2\n\t"
            "vsetivli zero, 16, e32, m4\n\t"
            "vredsum.vs v8, v4, v8\n\t"
            "vmv.x.s %[vsums], v8"
            : [tmp] "=&r" (tmp), [vsums] "=&r" (vsums)
            : [sc] "r" (sc), [scale] "r" (atmp), [bsums] "r" (y[i].bsums)
            : "memory"
            , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
            , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
            , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
            , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
        );
        sumf += dmin * vsums;
        ಪೂರ್ಣಾಂಕ isum = 0;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K/128; ++j) {
            __asm__ __volatile__(
                "lb zero, 31(%[q2])\n\t"
                "addi %[tmp], %[q2], 16\n\t"
                "addi %[t1], %[q8], 16\n\t"
                "vsetivli zero, 16, e8, m1\n\t"
                "vle8.v v0, (%[q2])\n\t"
                "vle8.v v1, (%[tmp])\n\t"
                "vsrl.vi v2, v0, 2\n\t"
                "vsrl.vi v3, v1, 2\n\t"
                "vsrl.vi v4, v0, 4\n\t"
                "addi %[tmp], %[q8], 32\n\t"
                "vle8.v v8, (%[q8])\n\t"
                "vle8.v v9, (%[t1])\n\t"
                "addi %[t1], %[t1], 32\n\t"
                "vsrl.vi v5, v1, 4\n\t"
                "vsrl.vi v6, v0, 6\n\t"
                "vsrl.vi v7, v1, 6\n\t"
                "vle8.v v10, (%[tmp])\n\t"
                "vle8.v v11, (%[t1])\n\t"
                "addi %[tmp], %[tmp], 32\n\t"
                "addi %[t1], %[t1], 32\n\t"
                "vand.vi v0, v0, 0x3\n\t"
                "vand.vi v1, v1, 0x3\n\t"
                "vand.vi v2, v2, 0x3\n\t"
                "vle8.v v12, (%[tmp])\n\t"
                "vle8.v v13, (%[t1])\n\t"
                "addi %[tmp], %[tmp], 32\n\t"
                "addi %[t1], %[t1], 32\n\t"
                "vand.vi v3, v3, 0x3\n\t"
                "vand.vi v4, v4, 0x3\n\t"
                "vand.vi v5, v5, 0x3\n\t"
                "vle8.v v14, (%[tmp])\n\t"
                "vle8.v v15, (%[t1])\n\t"
                "vwmul.vv v16, v0, v8\n\t"
                "vwmul.vv v18, v1, v9\n\t"
                "vwmul.vv v20, v2, v10\n\t"
                "vwmul.vv v22, v3, v11\n\t"
                "vwmul.vv v24, v4, v12\n\t"
                "vwmul.vv v26, v5, v13\n\t"
                "vwmul.vv v28, v6, v14\n\t"
                "vwmul.vv v30, v7, v15\n\t"
                "vsetivli zero, 8, e16, m1\n\t"
                "vmv.v.x v0, zero\n\t"
                "lbu %[tmp], 0(%[scale])\n\t"
                "vwredsum.vs v8, v16, v0\n\t"
                "vwredsum.vs v9, v18, v0\n\t"
                "lbu %[t1], 1(%[scale])\n\t"
                "vwredsum.vs v10, v20, v0\n\t"
                "vwredsum.vs v11, v22, v0\n\t"
                "lbu %[t2], 2(%[scale])\n\t"
                "vwredsum.vs v12, v24, v0\n\t"
                "vwredsum.vs v13, v26, v0\n\t"
                "lbu %[t3], 3(%[scale])\n\t"
                "vwredsum.vs v14, v28, v0\n\t"
                "vwredsum.vs v15, v30, v0\n\t"
                "lbu %[t4], 4(%[scale])\n\t"
                "vwredsum.vs v8, v17, v8\n\t"
                "vwredsum.vs v9, v19, v9\n\t"
                "lbu %[t5], 5(%[scale])\n\t"
                "vwredsum.vs v10, v21, v10\n\t"
                "vwredsum.vs v11, v23, v11\n\t"
                "lbu %[t6], 6(%[scale])\n\t"
                "vwredsum.vs v12, v25, v12\n\t"
                "vwredsum.vs v13, v27, v13\n\t"
                "lbu %[t7], 7(%[scale])\n\t"
                "vwredsum.vs v14, v29, v14\n\t"
                "vwredsum.vs v15, v31, v15\n\t"
                "vsetivli zero, 4, e32, m1\n\t"
                "vmul.vx v0, v8, %[tmp]\n\t"
                "vmul.vx v1, v9, %[t1]\n\t"
                "vmacc.vx v0, %[t2], v10\n\t"
                "vmacc.vx v1, %[t3], v11\n\t"
                "vmacc.vx v0, %[t4], v12\n\t"
                "vmacc.vx v1, %[t5], v13\n\t"
                "vmacc.vx v0, %[t6], v14\n\t"
                "vmacc.vx v1, %[t7], v15\n\t"
                "vmv.x.s %[tmp], v0\n\t"
                "vmv.x.s %[t1], v1\n\t"
                "add %[isum], %[isum], %[tmp]\n\t"
                "add %[isum], %[isum], %[t1]"
                : [tmp] "=&r" (tmp), [t1] "=&r" (t1), [t2] "=&r" (t2), [t3] "=&r" (t3)
                , [t4] "=&r" (t4), [t5] "=&r" (t5), [t6] "=&r" (t6), [t7] "=&r" (t7)
                , [isum] "+&r" (isum)
                : [q2] "r" (q2), [scale] "r" (patmp), [q8] "r" (q8)
                : "memory"
                , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );
            q2 += 32; q8 += 128; patmp += 8;
        }

        sumf += dall * isum;
    }

    *s = sumf;
}

ಶೂನ್ಯ ggml_vec_dot_q2_K_q8_K_vl256(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_K * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    ದಶಮಾಂಶ sumf = 0;
    uint8_t atmp[16];

    uint8_t ತಾತ್ಕಾಲಿಕ_01[32] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ uint8_t * q2 = x[i].qs;
        ಸ್ಥಿರಾಂಕ int8_t *  q8 = y[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t * sc = x[i].scales;

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dall = y[i].d * GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin = -y[i].d * GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].dmin);

        size_t vl = 16;

        vuint8m1_t scales = __riscv_vle8_v_u8m1(sc, vl);
        vuint8m1_t aux    = __riscv_vand_vx_u8m1(scales, 0x0F, vl);

        vint16m1_t q8sums = __riscv_vle16_v_i16m1(y[i].bsums, vl);

        vuint8mf2_t scales_2 = __riscv_vle8_v_u8mf2(sc, vl);
        vuint8mf2_t mins8    = __riscv_vsrl_vx_u8mf2(scales_2, 0x4, vl);
        vint16m1_t  mins     = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(mins8, vl));
        vint32m2_t  prod     = __riscv_vwmul_vv_i32m2(q8sums, mins, vl);
        vint32m1_t  vsums    = __riscv_vredsum_vs_i32m2_i32m1(prod, __riscv_vmv_v_x_i32m1(0, 1), vl);

        sumf += dmin * __riscv_vmv_x_s_i32m1_i32(vsums);

        vl = 32;

        vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
        vuint8m1_t v_b   = __riscv_vle8_v_u8m1(ತಾತ್ಕಾಲಿಕ_01, vl);

        uint8_t is   = 0;
        ಪೂರ್ಣಾಂಕ     isum = 0;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K / 128; ++j) {
            // load Q2
            vuint8m1_t q2_x = __riscv_vle8_v_u8m1(q2, vl);

            vuint8m1_t q2_0 = __riscv_vand_vx_u8m1(q2_x, 0x03, vl);
            vuint8m1_t q2_1 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q2_x, 0x2, vl), 0x03, vl);
            vuint8m1_t q2_2 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q2_x, 0x4, vl), 0x03, vl);
            vuint8m1_t q2_3 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q2_x, 0x6, vl), 0x03, vl);

            // duplicate scale elements for product
            vuint8m1_t sc0 = __riscv_vrgather_vv_u8m1(aux, __riscv_vadd_vx_u8m1(v_b, 0 + is, vl), vl);
            vuint8m1_t sc1 = __riscv_vrgather_vv_u8m1(aux, __riscv_vadd_vx_u8m1(v_b, 2 + is, vl), vl);
            vuint8m1_t sc2 = __riscv_vrgather_vv_u8m1(aux, __riscv_vadd_vx_u8m1(v_b, 4 + is, vl), vl);
            vuint8m1_t sc3 = __riscv_vrgather_vv_u8m1(aux, __riscv_vadd_vx_u8m1(v_b, 6 + is, vl), vl);

            vint16m2_t p0 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(q2_0, sc0, vl));
            vint16m2_t p1 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(q2_1, sc1, vl));
            vint16m2_t p2 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(q2_2, sc2, vl));
            vint16m2_t p3 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(q2_3, sc3, vl));

            // load Q8
            vint8m1_t q8_0 = __riscv_vle8_v_i8m1(q8, vl);
            vint8m1_t q8_1 = __riscv_vle8_v_i8m1(q8 + 32, vl);
            vint8m1_t q8_2 = __riscv_vle8_v_i8m1(q8 + 64, vl);
            vint8m1_t q8_3 = __riscv_vle8_v_i8m1(q8 + 96, vl);

            vint32m4_t s0 = __riscv_vwmul_vv_i32m4(p0, __riscv_vwcvt_x_x_v_i16m2(q8_0, vl), vl);
            vint32m4_t s1 = __riscv_vwmul_vv_i32m4(p1, __riscv_vwcvt_x_x_v_i16m2(q8_1, vl), vl);
            vint32m4_t s2 = __riscv_vwmul_vv_i32m4(p2, __riscv_vwcvt_x_x_v_i16m2(q8_2, vl), vl);
            vint32m4_t s3 = __riscv_vwmul_vv_i32m4(p3, __riscv_vwcvt_x_x_v_i16m2(q8_3, vl), vl);

            vint32m1_t isum0 = __riscv_vredsum_vs_i32m4_i32m1(__riscv_vadd_vv_i32m4(s0, s1, vl), vzero, vl);
            vint32m1_t isum1 = __riscv_vredsum_vs_i32m4_i32m1(__riscv_vadd_vv_i32m4(s2, s3, vl), isum0, vl);

            isum += __riscv_vmv_x_s_i32m1_i32(isum1);

            q2 += 32;
            q8 += 128;
            is = 8;
        }

        sumf += dall * isum;
    }

    *s = sumf;
}
#ಆದರೆ_ಅಂತ್ಯ

ಶೂನ್ಯ ggml_vec_dot_q2_K_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
#ಆದರೆ defined __riscv_xtheadvector
    ggml_vec_dot_q2_K_q8_K_xtheadvector(n, s, bs, vx, bx, vy, by, nrc);
#ಅಥವಾಆದರೆ defined __riscv_v
    ಆಯ್ಕೆ (__riscv_vlenb() * 8) {
        ಸ್ಥಿತಿ 128:
            ggml_vec_dot_q2_K_q8_K_vl128(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ggml_vec_dot_q2_K_q8_K_vl256(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ggml_vec_dot_q2_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

#ಆದರೆ defined __riscv_xtheadvector
ಶೂನ್ಯ ggml_vec_dot_q3_K_q8_K_xtheadvector(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ uint32_t kmask1 = 0x03030303;
    ಸ್ಥಿರಾಂಕ uint32_t kmask2 = 0x0f0f0f0f;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q3_K * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    uint32_t utmp[4];
    ದಶಮಾಂಶ sumf = 0;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ q3 = x[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ qh = x[i].hmask;
        ಸ್ಥಿರಾಂಕ  int8_t * ಸೀಮಿತ q8 = y[i].qs;

        int8_t * scale = (int8_t *)utmp;
        ಪೂರ್ಣಾಂಕ tmp;
        __asm__ __volatile__(
            "li %[tmp], 12\n\t"
            "th.vsetvli zero, %[tmp], e8, m1\n\t"
            "th.vlb.v v0, (%[s6b])\n\t"
            "th.vmv.v.v v2, v0\n\t"
            "li %[tmp], 2\n\t"
            "th.vsetvli zero, %[tmp], e64, m1\n\t"
            "th.vmv.v.x v9, %[sh]\n\t"\
            "th.vslidedown.vi v1, v0, 1\n\t"
            "th.vslide1up.vx v8, v9, zero\n\t" // {0, 0, 4, 4}
            "th.vslideup.vi v0, v2, 1\n\t" // {aux[0], aux[1], aux[0], aux[1]}
            "li %[tmp], 4\n\t"
            "th.vsetvli zero, %[tmp], e32, m1\n\t"
            "th.vid.v v9\n\t"
            "th.vmv.x.s %[tmp], v1\n\t"
            "th.vsll.vi v9, v9, 1\n\t" // {0, 2, 4, 6}
            "th.vmv.v.x v1, %[tmp]\n\t" // {aux[2], aux[2], aux[2], aux[2]}
            "th.vsrl.vv v4, v1, v9\n\t"
            "th.vsrl.vv v2, v0, v8\n\t"
            "th.vand.vx v5, v4, %[kmask1]\n\t"
            "th.vand.vx v3, v2, %[kmask2]\n\t"
            "th.vsll.vi v6, v5, 4\n\t"
            "th.vor.vv v7, v6, v3\n\t"
            "li %[tmp], 16\n\t"
            "th.vsetvli zero, %[tmp], e8, m1\n\t"
            "th.vsub.vx v0, v7, %[c]\n\t"
            "th.vsb.v v0, (%[scale])"
            : [tmp] "=&r" (tmp)
            : [sh] "r" (0x0000000400000004), [s6b] "r" (x[i].scales), [c] "r" (32)
            , [scale] "r" (scale), [kmask1] "r" (kmask1), [kmask2] "r" (kmask2)
            : "memory"
            , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
            , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
            , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
            , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
        );

        uint8_t m = 1;
        ಪೂರ್ಣಾಂಕ isum = 0;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K; j += 128) {
            __asm__ __volatile__(
                // fixme: use v0p7 mask layout directly
                "th.vsetvli zero, %[vl32], e8, m2\n\t"
                "th.vlb.v v8, (%[q3])\n\t"
                "th.vsrl.vi v10, v8, 2\n\t"
                "th.vsrl.vi v12, v8, 4\n\t"
                "th.vsrl.vi v14, v8, 6\n\t"
                "th.vand.vi v8, v8, 3\n\t"
                "th.vand.vi v10, v10, 3\n\t"
                "th.vand.vi v12, v12, 3\n\t"
                "th.vlb.v v2, (%[qh])\n\t"
                "th.vand.vx v4, v2, %[m]\n\t"
                "slli %[m], %[m], 1\n\t"
                "th.vmseq.vx v0, v4, zero\n\t"
                "th.vadd.vi v8, v8, -4, v0.t\n\t"
                "th.vand.vx v4, v2, %[m]\n\t"
                "slli %[m], %[m], 1\n\t"
                "th.vmseq.vx v0, v4, zero\n\t"
                "th.vadd.vi v10, v10, -4, v0.t\n\t"
                "th.vand.vx v4, v2, %[m]\n\t"
                "slli %[m], %[m], 1\n\t"
                "th.vmseq.vx v0, v4, zero\n\t"
                "th.vadd.vi v12, v12, -4, v0.t\n\t"
                "th.vand.vx v4, v2, %[m]\n\t"
                "slli %[m], %[m], 1\n\t"
                "th.vmseq.vx v0, v4, zero\n\t"
                "th.vadd.vi v14, v14, -4, v0.t\n\t"
                "th.vsetvli zero, %[vl128], e8, m8\n\t"
                "th.vlb.v v0, (%[q8])\n\t"
                "th.vsetvli zero, %[vl64], e8, m4\n\t"
                "th.vwmul.vv v16, v0, v8\n\t"
                "th.vwmul.vv v24, v4, v12\n\t"
                "li %[tmp], 16\n\t"
                "th.vsetvli zero, %[tmp], e16, m2\n\t"
                "th.vmv.v.x v0, zero\n\t"
                "th.vwredsum.vs v10, v16, v0\n\t"
                "th.vwredsum.vs v9, v18, v0\n\t"
                "th.vwredsum.vs v8, v20, v0\n\t"
                "th.vwredsum.vs v7, v22, v0\n\t"
                "th.vwredsum.vs v11, v24, v0\n\t"
                "th.vwredsum.vs v12, v26, v0\n\t"
                "th.vwredsum.vs v13, v28, v0\n\t"
                "th.vwredsum.vs v14, v30, v0\n\t"
                "li %[tmp], 4\n\t"
                "th.vsetvli zero, %[tmp], e32, m1\n\t"
                "th.vslideup.vi v10, v9, 1\n\t"
                "th.vslideup.vi v8, v7, 1\n\t"
                "th.vslideup.vi v11, v12, 1\n\t"
                "th.vslideup.vi v13, v14, 1\n\t"
                "th.vslideup.vi v10, v8, 2\n\t"
                "th.vslideup.vi v11, v13, 2\n\t"
                "li %[tmp], 8\n\t"
                "th.vsetvli zero, %[tmp], e32, m2\n\t"
                "th.vlb.v v12, (%[scale])\n\t"
                "th.vmul.vv v10, v10, v12\n\t"
                "th.vredsum.vs v0, v10, v0\n\t"
                "th.vmv.x.s %[tmp], v0\n\t"
                "add %[isum], %[isum], %[tmp]"
                : [tmp] "=&r" (tmp), [m] "+&r" (m), [isum] "+&r" (isum)
                : [vl128] "r" (128), [vl64] "r" (64), [vl32] "r" (32)
                , [q3] "r" (q3), [qh] "r" (qh), [scale] "r" (scale), [q8] "r" (q8)
                : "memory"
                , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );
            q3 += 32;    q8 += 128;   scale += 8;
        }

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d;
        sumf += d * isum;
    }

    *s = sumf;
}
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined __riscv_v
ಶೂನ್ಯ ggml_vec_dot_q3_K_q8_K_vl128(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ uint32_t kmask1 = 0x03030303;
    ಸ್ಥಿರಾಂಕ uint32_t kmask2 = 0x0f0f0f0f;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q3_K * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    uint32_t utmp[4];
    ದಶಮಾಂಶ sumf = 0;
    uint32_t aux[3];

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ q3 = x[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ qh = x[i].hmask;
        ಸ್ಥಿರಾಂಕ  int8_t * ಸೀಮಿತ q8 = y[i].qs;

        int8_t * scale = (int8_t *)utmp;
        ಪೂರ್ಣಾಂಕ tmp, t1, t2, t3, t4, t5, t6, t7;
        __asm__ __volatile__(
            "vsetivli zero, 12, e8, m1\n\t"
            "vle8.v v0, (%[s6b])\n\t"
            "vmv1r.v v2, v0\n\t"
            "vsetivli zero, 2, e64, m1\n\t"
            "vmv.v.x v9, %[sh]\n\t"\
            "vslidedown.vi v1, v0, 1\n\t"
            "vslide1up.vx v8, v9, zero\n\t" // {0, 0, 4, 4}
            "vslideup.vi v0, v2, 1\n\t" // {aux[0], aux[1], aux[0], aux[1]}
            "vsetivli zero, 4, e32, m1\n\t"
            "vid.v v9\n\t"
            "vmv.x.s %[tmp], v1\n\t"
            "vsll.vi v9, v9, 1\n\t" // {0, 2, 4, 6}
            "vmv.v.x v1, %[tmp]\n\t" // {aux[2], aux[2], aux[2], aux[2]}
            "vsrl.vv v4, v1, v9\n\t"
            "vsrl.vv v2, v0, v8\n\t"
            "vand.vx v5, v4, %[kmask1]\n\t"
            "vand.vx v3, v2, %[kmask2]\n\t"
            "vsll.vi v6, v5, 4\n\t"
            "vor.vv v7, v6, v3\n\t"
            "vsetivli zero, 16, e8, m1\n\t"
            "vsub.vx v0, v7, %[c]\n\t"
            "vse8.v v0, (%[scale])"
            : [tmp] "=&r" (tmp)
            : [sh] "r" (0x0000000400000004), [s6b] "r" (x[i].scales), [c] "r" (32)
            , [scale] "r" (scale), [kmask1] "r" (kmask1), [kmask2] "r" (kmask2)
            : "memory"
            , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
            , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
            , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
            , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
        );

        uint8_t m = 1;
        ಪೂರ್ಣಾಂಕ isum = 0;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K; j += 128) {
            __asm__ __volatile__(
                "lb zero, 31(%[q3])\n\t"
                "vsetvli zero, %[vl32], e8, m2, ta, mu\n\t"
                "vle8.v v8, (%[q3])\n\t"
                "vsrl.vi v10, v8, 2\n\t"
                "vsrl.vi v12, v8, 4\n\t"
                "vsrl.vi v14, v8, 6\n\t"
                "lb zero, 64(%[q8])\n\t"
                "vand.vi v8, v8, 3\n\t"
                "vand.vi v10, v10, 3\n\t"
                "vand.vi v12, v12, 3\n\t"
                "vle8.v v2, (%[qh])\n\t"
                "lb zero, 127(%[q8])\n\t"
                "vand.vx v4, v2, %[m]\n\t"
                "slli %[m], %[m], 1\n\t"
                "vmseq.vx v0, v4, zero\n\t"
                "vadd.vi v8, v8, -4, v0.t\n\t"
                "lb zero, 0(%[q8])\n\t"
                "vand.vx v4, v2, %[m]\n\t"
                "slli %[m], %[m], 1\n\t"
                "vmseq.vx v0, v4, zero\n\t"
                "vadd.vi v10, v10, -4, v0.t\n\t"
                "vand.vx v4, v2, %[m]\n\t"
                "slli %[m], %[m], 1\n\t"
                "vmseq.vx v0, v4, zero\n\t"
                "vadd.vi v12, v12, -4, v0.t\n\t"
                "vand.vx v4, v2, %[m]\n\t"
                "slli %[m], %[m], 1\n\t"
                "vmseq.vx v0, v4, zero\n\t"
                "vadd.vi v14, v14, -4, v0.t\n\t"
                "vsetvli zero, %[vl128], e8, m8\n\t"
                "vle8.v v0, (%[q8])\n\t"
                "lb %[tmp], 0(%[scale])\n\t"
                "lb %[t1], 1(%[scale])\n\t"
                "lb %[t2], 2(%[scale])\n\t"
                "lb %[t3], 3(%[scale])\n\t"
                "vsetvli zero, %[vl64], e8, m4\n\t"
                "vwmul.vv v16, v0, v8\n\t"
                "vwmul.vv v24, v4, v12\n\t"
                "vsetivli zero, 16, e16, m2\n\t"
                "vmv.v.x v0, zero\n\t"
                "vwredsum.vs v8, v16, v0\n\t"
                "lb %[t4], 4(%[scale])\n\t"
                "lb %[t5], 5(%[scale])\n\t"
                "vwredsum.vs v9, v18, v0\n\t"
                "vwredsum.vs v10, v20, v0\n\t"
                "vwredsum.vs v11, v22, v0\n\t"
                "vwredsum.vs v12, v24, v0\n\t"
                "lb %[t6], 6(%[scale])\n\t"
                "lb %[t7], 7(%[scale])\n\t"
                "vwredsum.vs v13, v26, v0\n\t"
                "vwredsum.vs v14, v28, v0\n\t"
                "vwredsum.vs v15, v30, v0\n\t"
                "vsetivli zero, 4, e32, m1\n\t"
                "vmul.vx v0, v8, %[tmp]\n\t"
                "vmul.vx v1, v9, %[t1]\n\t"
                "vmacc.vx v0, %[t2], v10\n\t"
                "vmacc.vx v1, %[t3], v11\n\t"
                "vmacc.vx v0, %[t4], v12\n\t"
                "vmacc.vx v1, %[t5], v13\n\t"
                "vmacc.vx v0, %[t6], v14\n\t"
                "vmacc.vx v1, %[t7], v15\n\t"
                "vmv.x.s %[tmp], v0\n\t"
                "vmv.x.s %[t1], v1\n\t"
                "add %[isum], %[isum], %[tmp]\n\t"
                "add %[isum], %[isum], %[t1]"
                : [tmp] "=&r" (tmp), [t1] "=&r" (t1), [t2] "=&r" (t2), [t3] "=&r" (t3)
                , [t4] "=&r" (t4), [t5] "=&r" (t5), [t6] "=&r" (t6), [t7] "=&r" (t7)
                , [m] "+&r" (m), [isum] "+&r" (isum)
                : [vl128] "r" (128), [vl64] "r" (64), [vl32] "r" (32)
                , [q3] "r" (q3), [qh] "r" (qh), [scale] "r" (scale), [q8] "r" (q8)
                : "memory"
                , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );
            q3 += 32;    q8 += 128;   scale += 8;
        }

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d;
        sumf += d * isum;
    }

    *s = sumf;
}

ಶೂನ್ಯ ggml_vec_dot_q3_K_q8_K_vl256(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ uint32_t kmask1 = 0x03030303;
    ಸ್ಥಿರಾಂಕ uint32_t kmask2 = 0x0f0f0f0f;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q3_K * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;
    uint32_t utmp[4];
    ದಶಮಾಂಶ sumf = 0;
    uint32_t aux[3];

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT q3 = x[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT qh = x[i].hmask;
        ಸ್ಥಿರಾಂಕ  int8_t * GGML_RESTRICT q8 = y[i].qs;

        ಸ್ಮರಣೆ_ನಕಲು(aux, x[i].scales, 12);
        utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
        utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4);
        utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4);
        utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4);

        int8_t * scale = (int8_t *)utmp;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 16; ++j) scale[j] -= 32;


        size_t vl = 32;
        uint8_t m =  1;

        vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
        vuint8m1_t vqh = __riscv_vle8_v_u8m1(qh, vl);

        ಪೂರ್ಣಾಂಕ ಮೊತ್ತ_t = 0;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K; j += 128) {

            vl = 32;

            // load Q3
            vuint8m1_t q3_x = __riscv_vle8_v_u8m1(q3, vl);

            vint8m1_t q3_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(q3_x, 0x03, vl));
            vint8m1_t q3_1 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q3_x, 0x2, vl), 0x03 , vl));
            vint8m1_t q3_2 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q3_x, 0x4, vl), 0x03 , vl));
            vint8m1_t q3_3 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q3_x, 0x6, vl), 0x03 , vl));

            // compute mask for subtraction
            vuint8m1_t qh_m0 = __riscv_vand_vx_u8m1(vqh, m, vl);
            vbool8_t vmask_0 = __riscv_vmseq_vx_u8m1_b8(qh_m0, 0, vl);
            vint8m1_t q3_m0 = __riscv_vsub_vx_i8m1_mu(vmask_0, q3_0, q3_0, 0x4, vl);
            m <<= 1;

            vuint8m1_t qh_m1 = __riscv_vand_vx_u8m1(vqh, m, vl);
            vbool8_t vmask_1 = __riscv_vmseq_vx_u8m1_b8(qh_m1, 0, vl);
            vint8m1_t q3_m1 = __riscv_vsub_vx_i8m1_mu(vmask_1, q3_1, q3_1, 0x4, vl);
            m <<= 1;

            vuint8m1_t qh_m2 = __riscv_vand_vx_u8m1(vqh, m, vl);
            vbool8_t vmask_2 = __riscv_vmseq_vx_u8m1_b8(qh_m2, 0, vl);
            vint8m1_t q3_m2 = __riscv_vsub_vx_i8m1_mu(vmask_2, q3_2, q3_2, 0x4, vl);
            m <<= 1;

            vuint8m1_t qh_m3 = __riscv_vand_vx_u8m1(vqh, m, vl);
            vbool8_t vmask_3 = __riscv_vmseq_vx_u8m1_b8(qh_m3, 0, vl);
            vint8m1_t q3_m3 = __riscv_vsub_vx_i8m1_mu(vmask_3, q3_3, q3_3, 0x4, vl);
            m <<= 1;

            // load Q8 and take product with Q3
            vint16m2_t a0 = __riscv_vwmul_vv_i16m2(q3_m0, __riscv_vle8_v_i8m1(q8, vl), vl);
            vint16m2_t a1 = __riscv_vwmul_vv_i16m2(q3_m1, __riscv_vle8_v_i8m1(q8+32, vl), vl);
            vint16m2_t a2 = __riscv_vwmul_vv_i16m2(q3_m2, __riscv_vle8_v_i8m1(q8+64, vl), vl);
            vint16m2_t a3 = __riscv_vwmul_vv_i16m2(q3_m3, __riscv_vle8_v_i8m1(q8+96, vl), vl);

            vl = 16;

            // retrieve lane to multiply with scale
            vint32m2_t aux0_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a0, 0), (scale[0]), vl);
            vint32m2_t aux0_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a0, 1), (scale[1]), vl);
            vint32m2_t aux1_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a1, 0), (scale[2]), vl);
            vint32m2_t aux1_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a1, 1), (scale[3]), vl);
            vint32m2_t aux2_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a2, 0), (scale[4]), vl);
            vint32m2_t aux2_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a2, 1), (scale[5]), vl);
            vint32m2_t aux3_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a3, 0), (scale[6]), vl);
            vint32m2_t aux3_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a3, 1), (scale[7]), vl);

            vint32m1_t isum0 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(aux0_0, aux0_1, vl), vzero, vl);
            vint32m1_t isum1 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(aux1_0, aux1_1, vl), isum0, vl);
            vint32m1_t isum2 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(aux2_0, aux2_1, vl), isum1, vl);
            vint32m1_t isum3 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(aux3_0, aux3_1, vl), isum2, vl);

            ಮೊತ್ತ_t +=  __riscv_vmv_x_s_i32m1_i32(isum3);

            q3 += 32;    q8 += 128;   scale += 8;

        }

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d;

        sumf += d*ಮೊತ್ತ_t;

    }

    *s = sumf;
}

ಶೂನ್ಯ ggml_vec_dot_q3_K_q8_K_vl512(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ uint32_t kmask1 = 0x03030303;
    ಸ್ಥಿರಾಂಕ uint32_t kmask2 = 0x0f0f0f0f;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q3_K * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    // mask for processing 16 elements per prod register
    ಸ್ಥಿರಾಂಕ vuint16m1_t va_ಸೂಚ್ಯಂಕ = __riscv_vid_v_u16m1(32);
    ಸ್ಥಿರಾಂಕ vbool16_t va_ಮುಖವಾಡ = __riscv_vmsgtu_vx_u16m1_b16(va_ಸೂಚ್ಯಂಕ, 15, 32);

    uint32_t utmp[4];
    ದಶಮಾಂಶ sumf = 0;
    uint32_t aux[3];

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT q3 = x[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT qh = x[i].hmask;
        ಸ್ಥಿರಾಂಕ  int8_t * GGML_RESTRICT q8 = y[i].qs;

        ಸ್ಮರಣೆ_ನಕಲು(aux, x[i].scales, 12);
        utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
        utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4);
        utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4);
        utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4);

        int8_t * scale = (int8_t *)utmp;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 16; ++j) scale[j] -= 32;


        size_t vl = 32;
        uint8_t m =  1;

        vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
        vuint8mf2_t vqh = __riscv_vle8_v_u8mf2(qh, vl);

        ಪೂರ್ಣಾಂಕ ಮೊತ್ತ_t = 0;

        vint32m2_t vaux_0 = __riscv_vmv_v_x_i32m2(0, vl);
        vint32m2_t vaux_1 = __riscv_vmv_v_x_i32m2(0, vl);
        vint32m2_t vaux_2 = __riscv_vmv_v_x_i32m2(0, vl);
        vint32m2_t vaux_3 = __riscv_vmv_v_x_i32m2(0, vl);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K; j += 128) {

            vl = 32;

            // load Q3
            vuint8mf2_t q3_x = __riscv_vle8_v_u8mf2(q3, vl);

            vint8mf2_t q3_0 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vand_vx_u8mf2(q3_x, 0x03, vl));
            vint8mf2_t q3_1 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(q3_x, 0x2, vl), 0x03 , vl));
            vint8mf2_t q3_2 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(q3_x, 0x4, vl), 0x03 , vl));
            vint8mf2_t q3_3 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(q3_x, 0x6, vl), 0x03 , vl));

            // compute mask for subtraction
            vuint8mf2_t qh_m0 = __riscv_vand_vx_u8mf2(vqh, m, vl);
            vbool16_t vmask_0 = __riscv_vmseq_vx_u8mf2_b16(qh_m0, 0, vl);
            vint8mf2_t q3_m0 = __riscv_vsub_vx_i8mf2_mu(vmask_0, q3_0, q3_0, 0x4, vl);
            m <<= 1;

            vuint8mf2_t qh_m1 = __riscv_vand_vx_u8mf2(vqh, m, vl);
            vbool16_t vmask_1 = __riscv_vmseq_vx_u8mf2_b16(qh_m1, 0, vl);
            vint8mf2_t q3_m1 = __riscv_vsub_vx_i8mf2_mu(vmask_1, q3_1, q3_1, 0x4, vl);
            m <<= 1;

            vuint8mf2_t qh_m2 = __riscv_vand_vx_u8mf2(vqh, m, vl);
            vbool16_t vmask_2 = __riscv_vmseq_vx_u8mf2_b16(qh_m2, 0, vl);
            vint8mf2_t q3_m2 = __riscv_vsub_vx_i8mf2_mu(vmask_2, q3_2, q3_2, 0x4, vl);
            m <<= 1;

            vuint8mf2_t qh_m3 = __riscv_vand_vx_u8mf2(vqh, m, vl);
            vbool16_t vmask_3 = __riscv_vmseq_vx_u8mf2_b16(qh_m3, 0, vl);
            vint8mf2_t q3_m3 = __riscv_vsub_vx_i8mf2_mu(vmask_3, q3_3, q3_3, 0x4, vl);
            m <<= 1;

            // load Q8 and take product
            vint16m1_t va_q_0 = __riscv_vwmul_vv_i16m1(q3_m0, __riscv_vle8_v_i8mf2(q8, vl), vl);
            vint16m1_t va_q_1 = __riscv_vwmul_vv_i16m1(q3_m1, __riscv_vle8_v_i8mf2(q8+32, vl), vl);
            vint16m1_t va_q_2 = __riscv_vwmul_vv_i16m1(q3_m2, __riscv_vle8_v_i8mf2(q8+64, vl), vl);
            vint16m1_t va_q_3 = __riscv_vwmul_vv_i16m1(q3_m3, __riscv_vle8_v_i8mf2(q8+96, vl), vl);

            // accumulate
            vaux_0 = __riscv_vwmacc_vx_i32m2(vaux_0, scale[0], va_q_0, 16);
            vaux_1 = __riscv_vwmacc_vx_i32m2(vaux_1, scale[2], va_q_1, 16);
            vaux_2 = __riscv_vwmacc_vx_i32m2(vaux_2, scale[4], va_q_2, 16);
            vaux_3 = __riscv_vwmacc_vx_i32m2(vaux_3, scale[6], va_q_3, 16);
            //
            vaux_0 = __riscv_vwmacc_vx_i32m2_m(va_ಮುಖವಾಡ, vaux_0, scale[1], va_q_0, vl);
            vaux_1 = __riscv_vwmacc_vx_i32m2_m(va_ಮುಖವಾಡ, vaux_1, scale[3], va_q_1, vl);
            vaux_2 = __riscv_vwmacc_vx_i32m2_m(va_ಮುಖವಾಡ, vaux_2, scale[5], va_q_2, vl);
            vaux_3 = __riscv_vwmacc_vx_i32m2_m(va_ಮುಖವಾಡ, vaux_3, scale[7], va_q_3, vl);

            q3 += 32;    q8 += 128;   scale += 8;
        }

        vint32m1_t isum0 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vaux_0, vaux_1, vl), vzero, vl);
        vint32m1_t isum1 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vaux_2, vaux_3, vl), isum0, vl);

        ಮೊತ್ತ_t += __riscv_vmv_x_s_i32m1_i32(isum1);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d;

        sumf += d*ಮೊತ್ತ_t;
    }

    *s = sumf;
}

ಶೂನ್ಯ ggml_vec_dot_q3_K_q8_K_vl1024(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ uint32_t kmask1 = 0x03030303;
    ಸ್ಥಿರಾಂಕ uint32_t kmask2 = 0x0f0f0f0f;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q3_K * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    // mask for processing 16 elements per prod register
    ಸ್ಥಿರಾಂಕ vuint16mf2_t va_ಸೂಚ್ಯಂಕ = __riscv_vid_v_u16mf2(32);
    ಸ್ಥಿರಾಂಕ vbool32_t va_ಮುಖವಾಡ = __riscv_vmsgtu_vx_u16mf2_b32(va_ಸೂಚ್ಯಂಕ, 15, 32);

    uint32_t utmp[4];
    ದಶಮಾಂಶ sumf = 0;
    uint32_t aux[3];

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT q3 = x[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT qh = x[i].hmask;
        ಸ್ಥಿರಾಂಕ  int8_t * GGML_RESTRICT q8 = y[i].qs;

        ಸ್ಮರಣೆ_ನಕಲು(aux, x[i].scales, 12);
        utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
        utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4);
        utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4);
        utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4);

        int8_t * scale = (int8_t *)utmp;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 16; ++j) scale[j] -= 32;


        size_t vl = 32;
        uint8_t m =  1;

        vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
        vuint8mf4_t vqh = __riscv_vle8_v_u8mf4(qh, vl);

        ಪೂರ್ಣಾಂಕ ಮೊತ್ತ_t = 0;

        vint32m1_t vaux_0 = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t vaux_1 = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t vaux_2 = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t vaux_3 = __riscv_vmv_v_x_i32m1(0, vl);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K; j += 128) {

            vl = 32;

            // load Q3
            vuint8mf4_t q3_x = __riscv_vle8_v_u8mf4(q3, vl);

            vint8mf4_t q3_0 = __riscv_vreinterpret_v_u8mf4_i8mf4(__riscv_vand_vx_u8mf4(q3_x, 0x03, vl));
            vint8mf4_t q3_1 = __riscv_vreinterpret_v_u8mf4_i8mf4(__riscv_vand_vx_u8mf4(__riscv_vsrl_vx_u8mf4(q3_x, 0x2, vl), 0x03 , vl));
            vint8mf4_t q3_2 = __riscv_vreinterpret_v_u8mf4_i8mf4(__riscv_vand_vx_u8mf4(__riscv_vsrl_vx_u8mf4(q3_x, 0x4, vl), 0x03 , vl));
            vint8mf4_t q3_3 = __riscv_vreinterpret_v_u8mf4_i8mf4(__riscv_vand_vx_u8mf4(__riscv_vsrl_vx_u8mf4(q3_x, 0x6, vl), 0x03 , vl));

            // compute mask for subtraction
            vuint8mf4_t qh_m0 = __riscv_vand_vx_u8mf4(vqh, m, vl);
            vbool32_t vmask_0 = __riscv_vmseq_vx_u8mf4_b32(qh_m0, 0, vl);
            vint8mf4_t q3_m0 = __riscv_vsub_vx_i8mf4_mu(vmask_0, q3_0, q3_0, 0x4, vl);
            m <<= 1;

            vuint8mf4_t qh_m1 = __riscv_vand_vx_u8mf4(vqh, m, vl);
            vbool32_t vmask_1 = __riscv_vmseq_vx_u8mf4_b32(qh_m1, 0, vl);
            vint8mf4_t q3_m1 = __riscv_vsub_vx_i8mf4_mu(vmask_1, q3_1, q3_1, 0x4, vl);
            m <<= 1;

            vuint8mf4_t qh_m2 = __riscv_vand_vx_u8mf4(vqh, m, vl);
            vbool32_t vmask_2 = __riscv_vmseq_vx_u8mf4_b32(qh_m2, 0, vl);
            vint8mf4_t q3_m2 = __riscv_vsub_vx_i8mf4_mu(vmask_2, q3_2, q3_2, 0x4, vl);
            m <<= 1;

            vuint8mf4_t qh_m3 = __riscv_vand_vx_u8mf4(vqh, m, vl);
            vbool32_t vmask_3 = __riscv_vmseq_vx_u8mf4_b32(qh_m3, 0, vl);
            vint8mf4_t q3_m3 = __riscv_vsub_vx_i8mf4_mu(vmask_3, q3_3, q3_3, 0x4, vl);
            m <<= 1;

            // load Q8 and take product
            vint16mf2_t va_q_0 = __riscv_vwmul_vv_i16mf2(q3_m0, __riscv_vle8_v_i8mf4(q8, vl), vl);
            vint16mf2_t va_q_1 = __riscv_vwmul_vv_i16mf2(q3_m1, __riscv_vle8_v_i8mf4(q8+32, vl), vl);
            vint16mf2_t va_q_2 = __riscv_vwmul_vv_i16mf2(q3_m2, __riscv_vle8_v_i8mf4(q8+64, vl), vl);
            vint16mf2_t va_q_3 = __riscv_vwmul_vv_i16mf2(q3_m3, __riscv_vle8_v_i8mf4(q8+96, vl), vl);

            // accumulate
            vaux_0 = __riscv_vwmacc_vx_i32m1(vaux_0, scale[0], va_q_0, 16);
            vaux_1 = __riscv_vwmacc_vx_i32m1(vaux_1, scale[2], va_q_1, 16);
            vaux_2 = __riscv_vwmacc_vx_i32m1(vaux_2, scale[4], va_q_2, 16);
            vaux_3 = __riscv_vwmacc_vx_i32m1(vaux_3, scale[6], va_q_3, 16);
            //
            vaux_0 = __riscv_vwmacc_vx_i32m1_m(va_ಮುಖವಾಡ, vaux_0, scale[1], va_q_0, vl);
            vaux_1 = __riscv_vwmacc_vx_i32m1_m(va_ಮುಖವಾಡ, vaux_1, scale[3], va_q_1, vl);
            vaux_2 = __riscv_vwmacc_vx_i32m1_m(va_ಮುಖವಾಡ, vaux_2, scale[5], va_q_2, vl);
            vaux_3 = __riscv_vwmacc_vx_i32m1_m(va_ಮುಖವಾಡ, vaux_3, scale[7], va_q_3, vl);

            q3 += 32;    q8 += 128;   scale += 8;
        }

        vint32m1_t isum0 = __riscv_vredsum_vs_i32m1_i32m1(__riscv_vadd_vv_i32m1(vaux_0, vaux_1, vl), vzero, vl);
        vint32m1_t isum1 = __riscv_vredsum_vs_i32m1_i32m1(__riscv_vadd_vv_i32m1(vaux_2, vaux_3, vl), isum0, vl);

        ಮೊತ್ತ_t += __riscv_vmv_x_s_i32m1_i32(isum1);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d;

        sumf += d*ಮೊತ್ತ_t;
    }

    *s = sumf;
}
#ಆದರೆ_ಅಂತ್ಯ

ಶೂನ್ಯ ggml_vec_dot_q3_K_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
#ಆದರೆ defined __riscv_xtheadvector
    ggml_vec_dot_q3_K_q8_K_xtheadvector(n, s, bs, vx, bx, vy, by, nrc);
#ಅಥವಾಆದರೆ defined __riscv_v
    ಆಯ್ಕೆ (__riscv_vlenb() * 8) {
        ಸ್ಥಿತಿ 128:
            ggml_vec_dot_q3_K_q8_K_vl128(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 256:
            ggml_vec_dot_q3_K_q8_K_vl256(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 512:
            ggml_vec_dot_q3_K_q8_K_vl512(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 1024:
            ggml_vec_dot_q3_K_q8_K_vl1024(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ggml_vec_dot_q3_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ggml_vec_dot_q3_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

#ಆದರೆ defined __riscv_xtheadvector
ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_q4_K_q8_K_xtheadvector(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint32_t kmask1 = 0x3f3f3f3f;
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint32_t kmask2 = 0x0f0f0f0f;
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    ಸ್ಥಿರಾಂಕ uint8_t * scales = (ಸ್ಥಿರಾಂಕ uint8_t*)&utmp[0];
    ಸ್ಥಿರಾಂಕ uint8_t * mins   = (ಸ್ಥಿರಾಂಕ uint8_t*)&utmp[2];

    ದಶಮಾಂಶ sumf = 0;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = y[i].d * GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin = y[i].d * GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].dmin);

        ಪೂರ್ಣಾಂಕ tmp, tmp2, sumi;
        __asm__ __volatile__(
            "li %[t1], 12\n\t"
            "th.vsetvli zero, %[t1], e8, m1\n\t"
            "th.vlb.v v1, (%[s6b])\n\t" // {aux[0], aux[1], aux[2]}
            "li %[t1], 4\n\t"
            "th.vsetvli zero, %[t1], e32, m1\n\t"
            "th.vslidedown.vi v2, v1, 2\n\t"
            "th.vmv.v.v v3, v2\n\t"
            "th.vslideup.vi v2, v3, 1\n\t" // {aux[2], aux[2]}
            "li %[t1], 2\n\t"
            "th.vsetvli zero, %[t1], e32, m1\n\t"
            "th.vmv.v.i v4, 4\n\t"
            "th.vand.vx v8, v1, %[kmask1]\n\t"
            "th.vslide1up.vx v5, v4, zero\n\t" // {0, 4}
            "th.vsrl.vi v6, v1, 6\n\t"
            "th.vsrl.vv v7, v2, v5\n\t"
            "th.vand.vx v0, v6, %[kmask3]\n\t"
            "th.vand.vx v2, v7, %[kmask2]\n\t"
            "th.vsll.vi v6, v0, 4\n\t"
            "li %[t2], 8\n\t"
            "addi %[t1], %[utmp], 4\n\t"
            "th.vor.vv v1, v6, v2\n\t"
            "th.vssw.v v8, (%[utmp]), %[t2]\n\t"
            "th.vssw.v v1, (%[t1]), %[t2]\n\t"
            "th.vsetvli zero, zero, e32, m2\n\t" // vl == 8
            "th.vlw.v v2, (%[bsums])\n\t"
            "th.vsetvli zero, %[t2], e16, m1\n\t"
            "th.vnsrl.vi v0, v2, 0\n\t"
            "th.vnsrl.vi v1, v2, 16\n\t"
            "th.vadd.vv v2, v0, v1\n\t"
            "th.vlbu.v v4, (%[mins])\n\t"
            "th.vwmul.vv v6, v4, v2\n\t"
            "th.vmv.v.x v0, zero\n\t"
            "th.vsetvli zero, %[t2], e32, m2\n\t"
            "th.vredsum.vs v0, v6, v0\n\t"
            "th.vmv.x.s %[sumi], v0"
            : [t1] "=&r" (tmp), [t2] "=&r" (tmp2), [sumi] "=&r" (sumi)
            : [bsums] "r" (y[i].bsums), [mins] "r" (mins), [utmp] "r" (utmp)
            , [s6b] "r" (x[i].scales), [kmask1] "r" (kmask1)
            , [kmask2] "r" (kmask2), [kmask3] "r" (kmask3)
            : "memory"
            , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
            , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
            , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
            , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
        );
        sumf -= dmin * sumi;

        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ q4 = x[i].qs;
        ಸ್ಥಿರಾಂಕ int8_t  * ಸೀಮಿತ q8 = y[i].qs;

        sumi = 0;
        ಸ್ಥಿರಾಂಕ uint8_t * scale = scales;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K/128; ++j) {
            ಪೂರ್ಣಾಂಕ vl128 = 128, vl64 = 64, vl32 = 32;
            __asm__ __volatile__(
                "th.vsetvli zero, %[vl128], e8, m8\n\t"
                "th.vlb.v v8, (%[q8])\n\t"
                "th.vsetvli zero, %[vl64], e8, m4\n\t"
                "th.vlb.v v0, (%[q4])\n\t"
                "th.vsrl.vi v4, v0, 4\n\t"
                "th.vand.vi v0, v0, 0xF\n\t"
                "th.vsetvli zero, %[vl32], e8, m2\n\t"
                "th.vwmul.vv v28, v6, v14\n\t"
                "th.vwmul.vv v20, v4, v10\n\t"
                "th.vwmul.vv v24, v2, v12\n\t"
                "th.vwmul.vv v16, v0, v8\n\t"
                "li %[tmp], 4\n\t"
                "th.vsetvli zero, %[tmp], e32, m1\n\t"
                "th.vlbu.v v1, (%[scale])\n\t"
                "th.vmv.v.x v0, zero\n\t"
                "th.vsetvli zero, %[vl32], e16, m4\n\t"
                "th.vwredsum.vs v6, v24, v0\n\t"
                "th.vwredsum.vs v7, v28, v0\n\t"
                "th.vwredsum.vs v4, v16, v0\n\t"
                "th.vwredsum.vs v5, v20, v0\n\t"
                "th.vsetvli zero, %[tmp], e32, m1\n\t"
                "th.vslideup.vi v6, v7, 1\n\t"
                "th.vslideup.vi v4, v5, 1\n\t"
                "th.vslideup.vi v4, v6, 2\n\t"
                "th.vmul.vv v8, v4, v1\n\t"
                "th.vredsum.vs v0, v8, v0\n\t"
                "th.vmv.x.s %[tmp], v0\n\t"
                "add %[sumi], %[sumi], %[tmp]"
                : [tmp] "=&r" (tmp), [sumi] "+&r" (sumi)
                : [vl128] "r" (vl128), [vl64] "r" (vl64), [vl32] "r" (vl32)
                , [q4] "r" (q4), [q8] "r" (q8), [scale] "r" (scale)
                : "memory"
                , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );

            q4 += 64;    q8 += 128;    scale += 4;
        }

        sumf += d * sumi;

    }

    *s = sumf;
}
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined __riscv_v
ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_q4_K_q8_K_vl128(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint32_t kmask1 = 0x3f3f3f3f;
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint32_t kmask2 = 0x0f0f0f0f;
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    ಸ್ಥಿರಾಂಕ uint8_t * scales = (ಸ್ಥಿರಾಂಕ uint8_t*)&utmp[0];
    ಸ್ಥಿರಾಂಕ uint8_t * mins   = (ಸ್ಥಿರಾಂಕ uint8_t*)&utmp[2];

    ದಶಮಾಂಶ sumf = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = y[i].d * GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin = y[i].d * GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].dmin);

        ದಶಮಾಂಶ ftmp, ft2;
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ q40;
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ q41;
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ q42;
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ q43;
        ಸ್ಥಿರಾಂಕ int8_t  * ಸೀಮಿತ q80;
        ಸ್ಥಿರಾಂಕ int8_t  * ಸೀಮಿತ q81;
        ಸ್ಥಿರಾಂಕ int8_t  * ಸೀಮಿತ q82;
        ಸ್ಥಿರಾಂಕ int8_t  * ಸೀಮಿತ q83;
        ಪೂರ್ಣಾಂಕ s0, s1, s2, s3;

        __asm__ __volatile__(
            "li %[s1], 8\n\t"
            "vsetivli zero, 4, e32, m1, ta, ma\n\t"
            "vle32.v v1, (%[s6b])\n\t"
            "vslide1down.vx v1, v1, zero\n\t"
            "vmv.v.x v16, zero\n\t"
            "vslidedown.vi v2, v1, 2\n\t"
            "vmv1r.v v3, v2\n\t"
            "vslideup.vi v2, v3, 1\n\t" // {aux[2], aux[2]}
            "vsetivli zero, 2, e32, m1, ta, ma\n\t"
            "vmv.v.i v4, 4\n\t"
            "vand.vx v8, v1, %[kmask1]\n\t"
            "vslide1up.vx v5, v4, zero\n\t" // {0, 4}
            "vsrl.vi v6, v1, 6\n\t"
            "vsrl.vv v7, v2, v5\n\t"
            "vsse32.v v8, (%[utmp]), %[s1]\n\t"
            "vand.vx v0, v6, %[kmask3]\n\t"
            "vand.vx v2, v7, %[kmask2]\n\t"
            "vsll.vi v6, v0, 4\n\t"
            "addi %[s0], %[utmp], 4\n\t"
            "vor.vv v1, v6, v2\n\t"
            "vsse32.v v1, (%[s0]), %[s1]\n\t"
            "vsetivli zero, 8, e16, m1, ta, ma\n\t"
            "vle32.v v2, (%[bsums])\n\t"
            "vnsrl.wi v0, v2, 0\n\t"
            "vnsrl.wi v1, v2, 16\n\t"
            "vadd.vv v2, v0, v1\n\t"
            "vle8.v v3, (%[mins])\n\t"
            "vzext.vf2 v4, v3\n\t"
            "vwmul.vv v6, v4, v2\n\t"
            "vsetivli zero, 4, e32, m1, ta, ma\n\t"
            "vredsum.vs v0, v6, v16\n\t"
            "vredsum.vs v0, v7, v0\n\t"
            "vfcvt.f.x.v v0, v0\n\t"
            "vfmv.f.s %[ftmp], v0\n\t"
            "vsetivli zero, 16, e8, m1, ta, ma\n\t"
            "vle8.v v0, (%[xs])\n\t"
            "fnmsub.s %[sumf], %[dmin], %[ftmp], %[sumf]\n\t"
            "addi %[q40], %[xs], 64\n\t"
            "addi %[q41], %[xs], 16\n\t"
            "addi %[q42], %[xs], 32\n\t"
            "addi %[q43], %[xs], 48\n\t"
            "addi %[q80], %[ys], 64\n\t"
            "vle8.v v1, (%[q41])\n\t"
            "vle8.v v2, (%[q42])\n\t"
            "addi %[q81], %[ys], 16\n\t"
            "addi %[q41], %[q41], 64\n\t"
            "addi %[q82], %[ys], 32\n\t"
            "vle8.v v3, (%[q43])\n\t"
            "vle8.v v8, (%[ys])\n\t"
            "addi %[q42], %[q42], 64\n\t"
            "addi %[q83], %[ys], 48\n\t"
            "addi %[q43], %[q43], 64\n\t"
            "vsrl.vi v4, v0, 4\n\t"
            "vle8.v v9, (%[q81])\n\t"
            "vle8.v v10, (%[q82])\n\t"
            "vand.vi v0, v0, 0xF\n\t"
            "addi %[q81], %[q81], 64\n\t"
            "vsrl.vi v5, v1, 4\n\t"
            "addi %[q82], %[q82], 64\n\t"
            "vle8.v v11, (%[q83])\n\t"
            "vle8.v v12, (%[q80])\n\t"
            "vand.vi v1, v1, 0xF\n\t"
            "addi %[q83], %[q83], 64\n\t"
            "vsrl.vi v6, v2, 4\n\t"
            "addi %[q80], %[q80], 64\n\t"
            "vle8.v v13, (%[q81])\n\t"
            "vle8.v v14, (%[q82])\n\t"
            "vand.vi v2, v2, 0xF\n\t"
            "addi %[q81], %[q81], 64\n\t"
            "vsrl.vi v7, v3, 4\n\t"
            "addi %[q82], %[q82], 64\n\t"
            "vwmul.vv v16, v0, v8\n\t"
            "vle8.v v15, (%[q83])\n\t"
            "vle8.v v0, (%[q40])\n\t"
            "vand.vi v3, v3, 0xF\n\t"
            "addi %[q83], %[q83], 64\n\t"
            "vwmul.vv v24, v2, v12\n\t"
            "vwmul.vv v20, v4, v10\n\t"
            "vwmul.vv v28, v6, v14\n\t"
            "vwmacc.vv v16, v1, v9\n\t"
            "vle8.v v1, (%[q41])\n\t"
            "vle8.v v2, (%[q42])\n\t"
            "vwmacc.vv v24, v3, v13\n\t"
            "vwmacc.vv v20, v5, v11\n\t"
            "vwmacc.vv v28, v7, v15\n\t"
            "addi %[q40], %[q80], 64\n\t"
            "addi %[q41], %[q81], 64\n\t"
            "vle8.v v3, (%[q43])\n\t"
            "vle8.v v8, (%[q80])\n\t"
            "addi %[q42], %[q82], 64\n\t"
            "addi %[q43], %[q83], 64\n\t"
            "vsrl.vi v4, v0, 4\n\t"
            "vle8.v v9, (%[q81])\n\t"
            "vle8.v v10, (%[q82])\n\t"
            "vand.vi v0, v0, 0xF\n\t"
            "vsrl.vi v5, v1, 4\n\t"
            "vsrl.vi v7, v3, 4\n\t"
            "vand.vi v3, v3, 0xF\n\t"
            "vle8.v v11, (%[q83])\n\t"
            "vle8.v v12, (%[q40])\n\t"
            "vand.vi v1, v1, 0xF\n\t"
            "vsrl.vi v6, v2, 4\n\t"
            "vand.vi v2, v2, 0xF\n\t"
            "vwmul.vv v18, v0, v8\n\t"
            "vle8.v v13, (%[q41])\n\t"
            "vle8.v v14, (%[q42])\n\t"
            "vwmul.vv v26, v2, v12\n\t"
            "vwmul.vv v22, v4, v10\n\t"
            "vwmul.vv v30, v6, v14\n\t"
            "vwmacc.vv v18, v1, v9\n\t"
            "vle8.v v15, (%[q43])\n\t"
            "vwmacc.vv v26, v3, v13\n\t"
            "vwmacc.vv v22, v5, v11\n\t"
            "vwmacc.vv v30, v7, v15\n\t"
            "vmv.v.x v0, zero\n\t"
            "vsetivli zero, 16, e16, m2, ta, ma\n\t"
            "vwredsum.vs v4, v16, v0\n\t"
            "lbu %[s0], 0(%[scale])\n\t"
            "vwredsum.vs v5, v20, v0\n\t"
            "lbu %[s1], 1(%[scale])\n\t"
            "vwredsum.vs v6, v24, v0\n\t"
            "lbu %[s2], 2(%[scale])\n\t"
            "vwredsum.vs v7, v28, v0\n\t"
            "lbu %[s3], 3(%[scale])\n\t"
            "vwredsum.vs v8, v18, v0\n\t"
            "lbu %[q40], 4(%[scale])\n\t"
            "vwredsum.vs v9, v22, v0\n\t"
            "lbu %[q41], 5(%[scale])\n\t"
            "vwredsum.vs v10, v26, v0\n\t"
            "lbu %[q42], 6(%[scale])\n\t"
            "vwredsum.vs v11, v30, v0\n\t"
            "lbu %[q43], 7(%[scale])\n\t"
            "vsetivli zero, 4, e32, m1, ta, ma\n\t"
            "vmul.vx v0, v4, %[s0]\n\t"
            "vmul.vx v1, v8, %[q40]\n\t"
            "vmacc.vx v0, %[s1], v5\n\t"
            "vmacc.vx v1, %[q41], v9\n\t"
            "vmacc.vx v0, %[s2], v6\n\t"
            "vmacc.vx v1, %[q42], v10\n\t"
            "vmacc.vx v0, %[s3], v7\n\t"
            "vmacc.vx v1, %[q43], v11\n\t"
            "vfcvt.f.x.v v0, v0\n\t"
            "vfcvt.f.x.v v1, v1\n\t"
            "vfmv.f.s %[ft2], v0\n\t"
            "vfmv.f.s %[ftmp], v1\n\t"
            "fadd.s %[ft2], %[ft2], %[ftmp]\n\t"
            "fmadd.s %[sumf], %[d], %[ft2], %[sumf]"
            : [ftmp] "=&f" (ftmp), [sumf] "+&f" (sumf), [ft2] "=&f" (ft2)
            , [s0] "=&r" (s0), [s1] "=&r" (s1), [s2] "=&r" (s2), [s3] "=&r" (s3)
            , [q40] "=&r" (q40), [q41] "=&r" (q41), [q42] "=&r" (q42), [q43] "=&r" (q43)
            , [q80] "=&r" (q80), [q81] "=&r" (q81), [q82] "=&r" (q82), [q83] "=&r" (q83)
            : [d] "f" (d), [ys] "r" (y[i].qs), [xs] "r" (x[i].qs), [scale] "r" (scales)
            , [bsums] "r" (y[i].bsums), [mins] "r" (mins), [utmp] "r" (utmp)
            , [s6b] "r" (&x[i]), [kmask1] "r" (kmask1), [dmin] "f" (dmin)
            , [kmask2] "r" (kmask2), [kmask3] "r" (kmask3)
            : "memory"
            , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
            , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
            , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
            , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
        );
    }

    *s = sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_q4_K_q8_K_vl256(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint32_t kmask1 = 0x3f3f3f3f;
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint32_t kmask2 = 0x0f0f0f0f;
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    ಸ್ಥಿರಾಂಕ uint8_t * scales = (ಸ್ಥಿರಾಂಕ uint8_t*)&utmp[0];
    ಸ್ಥಿರಾಂಕ uint8_t * mins   = (ಸ್ಥಿರಾಂಕ uint8_t*)&utmp[2];

    ದಶಮಾಂಶ sumf = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        size_t vl = 8;

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = y[i].d * GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin = y[i].d * GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].dmin);

        vint16mf2_t q8sums_0 = __riscv_vlse16_v_i16mf2(y[i].bsums, 4, vl);
        vint16mf2_t q8sums_1 = __riscv_vlse16_v_i16mf2(y[i].bsums+1, 4, vl);
        vint16mf2_t q8sums   = __riscv_vadd_vv_i16mf2(q8sums_0, q8sums_1, vl);

        ಸ್ಮರಣೆ_ನಕಲು(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        ಸ್ಥಿರಾಂಕ uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        vuint8mf4_t mins8  = __riscv_vle8_v_u8mf4(mins, vl);
        vint16mf2_t v_mins = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vzext_vf2_u16mf2(mins8, vl));
        vint32m1_t  prod   = __riscv_vwmul_vv_i32m1(q8sums, v_mins, vl);

        vint32m1_t sumi = __riscv_vredsum_vs_i32m1_i32m1(prod, __riscv_vmv_v_x_i32m1(0, 1), vl);
        sumf -= dmin * __riscv_vmv_x_s_i32m1_i32(sumi);

        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT q4 = x[i].qs;
        ಸ್ಥಿರಾಂಕ int8_t  * GGML_RESTRICT q8 = y[i].qs;

        vl = 32;

        int32_t ಮೊತ್ತ_1 = 0;
        int32_t ಮೊತ್ತ_2 = 0;

        vint16m1_t vzero = __riscv_vmv_v_x_i16m1(0, 1);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K/64; ++j) {
            // load Q4
            vuint8m1_t q4_x = __riscv_vle8_v_u8m1(q4, vl);

            // load Q8 and multiply it with lower Q4 nibble
            vint8m1_t  q8_0 = __riscv_vle8_v_i8m1(q8, vl);
            vint8m1_t  q4_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(q4_x, 0x0F, vl));
            vint16m2_t qv_0 = __riscv_vwmul_vv_i16m2(q4_0, q8_0, vl);
            vint16m1_t vs_0 = __riscv_vredsum_vs_i16m2_i16m1(qv_0, vzero, vl);

            ಮೊತ್ತ_1 += __riscv_vmv_x_s_i16m1_i16(vs_0) * scales[2*j+0];

            // load Q8 and multiply it with upper Q4 nibble
            vint8m1_t  q8_1 = __riscv_vle8_v_i8m1(q8+32, vl);
            vint8m1_t  q4_1 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(q4_x, 0x04, vl));
            vint16m2_t qv_1 = __riscv_vwmul_vv_i16m2(q4_1, q8_1, vl);
            vint16m1_t vs_1 = __riscv_vredsum_vs_i16m2_i16m1(qv_1, vzero, vl);

            ಮೊತ್ತ_2 += __riscv_vmv_x_s_i16m1_i16(vs_1) * scales[2*j+1];

            q4 += 32;    q8 += 64;

        }

        sumf += d*(ಮೊತ್ತ_1 + ಮೊತ್ತ_2);

    }

    *s = sumf;
}
#ಆದರೆ_ಅಂತ್ಯ

ಶೂನ್ಯ ggml_vec_dot_q4_K_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
#ಆದರೆ defined __riscv_xtheadvector
    ggml_vec_dot_q4_K_q8_K_xtheadvector(n, s, bs, vx, bx, vy, by, nrc);
#ಅಥವಾಆದರೆ defined __riscv_v
    ಆಯ್ಕೆ (__riscv_vlenb() * 8) {
        ಸ್ಥಿತಿ 128:
            ggml_vec_dot_q4_K_q8_K_vl128(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: // 256 and above
            ggml_vec_dot_q4_K_q8_K_vl256(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ggml_vec_dot_q4_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_vec_dot_q5_K_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy,  size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint32_t kmask1 = 0x3f3f3f3f;
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint32_t kmask2 = 0x0f0f0f0f;
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#ಆದರೆ defined __riscv_v

    ಸ್ಥಿರಾಂಕ uint8_t * scales = (ಸ್ಥಿರಾಂಕ uint8_t*)&utmp[0];
    ಸ್ಥಿರಾಂಕ uint8_t * mins   = (ಸ್ಥಿರಾಂಕ uint8_t*)&utmp[2];

    ದಶಮಾಂಶ sumf = 0;
    ದಶಮಾಂಶ sums = 0.0;

    size_t vl;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {

        vl = 8;

        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT q5 = x[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT hm = x[i].qh;
        ಸ್ಥಿರಾಂಕ  int8_t * GGML_RESTRICT q8 = y[i].qs;

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].dmin) * y[i].d;

        vint16m1_t q8sums_0 = __riscv_vlse16_v_i16m1(y[i].bsums, 4, vl);
        vint16m1_t q8sums_1 = __riscv_vlse16_v_i16m1(y[i].bsums+1, 4, vl);
        vint16m1_t q8sums = __riscv_vadd_vv_i16m1(q8sums_0, q8sums_1, vl);

        ಸ್ಮರಣೆ_ನಕಲು(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        ಸ್ಥಿರಾಂಕ uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        vuint8mf2_t mins8 = __riscv_vle8_v_u8mf2(mins, vl);
        vint16m1_t v_mins = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(mins8, vl));
        vint32m2_t prod = __riscv_vwmul_vv_i32m2(q8sums, v_mins, vl);

        vint32m1_t sumi = __riscv_vredsum_vs_i32m2_i32m1(prod, __riscv_vmv_v_x_i32m1(0, 1), vl);
        sumf -= dmin * __riscv_vmv_x_s_i32m1_i32(sumi);

        vl = 32;
        int32_t aux32 = 0;
        ಪೂರ್ಣಾಂಕ is = 0;

        uint8_t m = 1;
        vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
        vuint8m2_t vqh = __riscv_vle8_v_u8m2(hm, vl);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K/64; ++j) {
            // load Q5 and Q8
            vuint8m2_t q5_x = __riscv_vle8_v_u8m2(q5, vl);
            vint8m2_t  q8_y1 = __riscv_vle8_v_i8m2(q8, vl);
            vint8m2_t  q8_y2 = __riscv_vle8_v_i8m2(q8+32, vl);

            // compute mask for addition
            vint8m2_t q5_a = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vand_vx_u8m2(q5_x, 0x0F, vl));
            vuint8m2_t qh_m1 = __riscv_vand_vx_u8m2(vqh, m, vl);
            vbool4_t vmask_1 = __riscv_vmsne_vx_u8m2_b4(qh_m1, 0, vl);
            vint8m2_t q5_m1 = __riscv_vadd_vx_i8m2_mu(vmask_1, q5_a, q5_a, 16, vl);
            m <<= 1;

            vint8m2_t q5_l = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vsrl_vx_u8m2(q5_x, 0x04, vl));
            vuint8m2_t qh_m2 = __riscv_vand_vx_u8m2(vqh, m, vl);
            vbool4_t vmask_2 = __riscv_vmsne_vx_u8m2_b4(qh_m2, 0, vl);
            vint8m2_t q5_m2 = __riscv_vadd_vx_i8m2_mu(vmask_2, q5_l, q5_l, 16, vl);
            m <<= 1;

            vint16m4_t v0 = __riscv_vwmul_vv_i16m4(q5_m1, q8_y1, vl);
            vint16m4_t v1 = __riscv_vwmul_vv_i16m4(q5_m2, q8_y2, vl);

            vint32m8_t vs1 = __riscv_vwmul_vx_i32m8(v0, scales[is++], vl);
            vint32m8_t vs2 = __riscv_vwmul_vx_i32m8(v1, scales[is++], vl);

            vint32m1_t vacc1 = __riscv_vredsum_vs_i32m8_i32m1(vs1, vzero, vl);
            vint32m1_t vacc2 = __riscv_vredsum_vs_i32m8_i32m1(vs2, vacc1, vl);

            aux32 += __riscv_vmv_x_s_i32m1_i32(vacc2);
            q5 += 32;    q8 += 64;
        }

        sums += aux32 * d;

    }

    *s = sumf+sums;

#ಇಲ್ಲದಿದ್ದರೆ

    ಬಳಸದ(x);
    ಬಳಸದ(y);
    ಬಳಸದ(kmask1);
    ಬಳಸದ(kmask2);
    ಬಳಸದ(kmask3);
    ಬಳಸದ(nb);
    ಬಳಸದ(utmp);

    ggml_vec_dot_q5_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

#ಆದರೆ defined __riscv_xtheadvector
ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_q6_K_q8_K_xtheadvector(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    ದಶಮಾಂಶ sumf = 0;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d;

        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ q6 = x[i].ql;
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ qh = x[i].qh;
        ಸ್ಥಿರಾಂಕ  int8_t * ಸೀಮಿತ q8 = y[i].qs;

        ಸ್ಥಿರಾಂಕ int8_t * ಸೀಮಿತ scale = x[i].scales;

        ಪೂರ್ಣಾಂಕ ಮೊತ್ತ_t = 0;
        ಪೂರ್ಣಾಂಕ t0;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K/128; ++j) {
            __asm__ __volatile__(
                "th.vsetvli zero, %[vl32], e8, m2\n\t" // vl == 32
                "th.vlb.v v4, (%[qh])\n\t"
                "th.vsll.vi v0, v4, 4\n\t"
                "th.vsll.vi v2, v4, 2\n\t"
                "th.vsrl.vi v6, v4, 2\n\t"
                "th.vsetvli zero, %[vl64], e8, m4\n\t" // vl == 64
                "th.vlb.v v8, (%[q6])\n\t"
                "th.vsrl.vi v12, v8, 4\n\t"
                "th.vand.vi v8, v8, 0xF\n\t"
                "th.vsetvli zero, %[vl128], e8, m8\n\t" // vl == 128
                "th.vand.vx v0, v0, %[mask]\n\t"
                "th.vor.vv v8, v8, v0\n\t"
                "th.vlb.v v0, (%[q8])\n\t"
                "th.vsub.vx v8, v8, %[vl32]\n\t"
                "th.vsetvli zero, %[vl64], e8, m4\n\t" // vl == 64
                "th.vwmul.vv v16, v0, v8\n\t"
                "th.vwmul.vv v24, v4, v12\n\t"
                "li %[t0], 16\n\t"
                "th.vsetvli zero, %[t0], e16, m2\n\t" // vl == 16
                "th.vmv.v.x v0, zero\n\t"
                "th.vwredsum.vs v10, v16, v0\n\t"
                "th.vwredsum.vs v9, v18, v0\n\t"
                "th.vwredsum.vs v8, v20, v0\n\t"
                "th.vwredsum.vs v7, v22, v0\n\t"
                "th.vwredsum.vs v11, v24, v0\n\t"
                "th.vwredsum.vs v12, v26, v0\n\t"
                "th.vwredsum.vs v13, v28, v0\n\t"
                "th.vwredsum.vs v14, v30, v0\n\t"
                "li %[t0], 4\n\t"
                "th.vsetvli zero, %[t0], e32, m1\n\t" // vl == 4
                "th.vslideup.vi v10, v9, 1\n\t"
                "th.vslideup.vi v8, v7, 1\n\t"
                "th.vslideup.vi v11, v12, 1\n\t"
                "th.vslideup.vi v13, v14, 1\n\t"
                "th.vslideup.vi v10, v8, 2\n\t"
                "th.vslideup.vi v11, v13, 2\n\t"
                "li %[t0], 8\n\t"
                "th.vsetvli zero, %[t0], e32, m2\n\t" // vl == 8
                "th.vlb.v v4, (%[scale])\n\t"
                "th.vmul.vv v2, v4, v10\n\t"
                "th.vredsum.vs v0, v2, v0\n\t"
                "th.vmv.x.s %[t0], v0\n\t"
                "add %[sumi], %[sumi], %[t0]"
                : [sumi] "+&r" (ಮೊತ್ತ_t), [t0] "=&r" (t0)
                : [qh] "r" (qh), [q6] "r" (q6), [q8] "r" (q8), [scale] "r" (scale)
                , [vl32] "r" (32), [vl64] "r" (64), [vl128] "r" (128)
                , [ಮುಖವಾಡ] "r" (0x30)
                : "memory"
                , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );
            q6 += 64;   qh += 32;   q8 += 128;   scale += 8;
        }

        sumf += d * ಮೊತ್ತ_t;

    }

    *s = sumf;
}
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined __riscv_v
ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_q6_K_q8_K_vl128(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    ದಶಮಾಂಶ sumf = 0.0f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        __builtin_prefetch(&x[i + 1].d, 0, 1);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d;

        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ q6 = x[i].ql;
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ qh = x[i].qh;
        ಸ್ಥಿರಾಂಕ  int8_t * ಸೀಮಿತ q8 = y[i].qs;

        ಸ್ಥಿರಾಂಕ int8_t * ಸೀಮಿತ scale = x[i].scales;

        ಪೂರ್ಣಾಂಕ q6h;
        ದಶಮಾಂಶ ftmp;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K/128; ++j) {
            __asm__ __volatile__(
                "addi %[q6h], %[q6], 32\n\t"
                "ld t0, 0(%[scale])\n\t"
                "addi %[scale], %[scale], 8\n\t"
                "slli t6, t0, 1 * 8\n\t"
                "lb zero, 0(%[q6])\n\t"
                "slli t5, t0, 2 * 8\n\t"
                "slli t4, t0, 3 * 8\n\t"
                "lb zero, 0(%[q6h])\n\t"
                "slli t3, t0, 4 * 8\n\t"
                "slli t2, t0, 5 * 8\n\t"
                "lb zero, 0(%[qh])\n\t"
                "lb zero, 31(%[q6h])\n\t"
                "slli t1, t0, 6 * 8\n\t"
                "srai a7, t0, 56\n\t"
                "vsetvli zero, %[vl32], e8, m2\n\t"
                "vle8.v v8, (%[q6])\n\t"
                "srai t6, t6, 56\n\t"
                "srai t5, t5, 56\n\t"
                "srai t4, t4, 56\n\t"
                "srai t3, t3, 56\n\t"
                "vle8.v v10, (%[q6h])\n\t"
                "addi %[q6], %[q6], 64\n\t"
                "slli t0, t0, 7 * 8\n\t"
                "srai t2, t2, 56\n\t"
                "srai t1, t1, 56\n\t"
                "srai t0, t0, 56\n\t"
                "vle8.v v4, (%[qh])\n\t"
                "vsrl.vi v12, v8, 4\n\t"
                "vsrl.vi v14, v10, 4\n\t"
                "lb zero, 0(%[q8])\n\t"
                "vand.vi v8, v8, 0xF\n\t"
                "vand.vi v10, v10, 0xF\n\t"
                "lb zero, 32(%[q8])\n\t"
                "vsll.vi v0, v4, 4\n\t"
                "vsll.vi v2, v4, 2\n\t"
                "lb zero, 64(%[q8])\n\t"
                "vsrl.vi v6, v4, 2\n\t"
                "vand.vx v0, v0, %[mask]\n\t"
                "lb zero, 96(%[q8])\n\t"
                "vand.vx v2, v2, %[mask]\n\t"
                "vand.vx v4, v4, %[mask]\n\t"
                "vand.vx v6, v6, %[mask]\n\t"
                "vor.vv v8, v8, v0\n\t"
                "lb zero, 127(%[q8])\n\t"
                "vor.vv v10, v10, v2\n\t"
                "vor.vv v12, v12, v4\n\t"
                "vor.vv v14, v14, v6\n\t"
                "vsetvli zero, %[vl128], e8, m8\n\t"
                "vle8.v v0, (%[q8])\n\t"
                "vsub.vx v8, v8, %[vl32]\n\t"
                "vsetvli zero, %[vl64], e8, m4\n\t"
                "vwmul.vv v16, v0, v8\n\t"
                "vwmul.vv v24, v4, v12\n\t"
                "vsetivli zero, 16, e16, m2\n\t"
                "vmv.v.x v0, zero\n\t"
                "vwredsum.vs v10, v16, v0\n\t"
                "vwredsum.vs v9, v18, v0\n\t"
                "vwredsum.vs v8, v20, v0\n\t"
                "vwredsum.vs v7, v22, v0\n\t"
                "vwredsum.vs v11, v24, v0\n\t"
                "vwredsum.vs v12, v26, v0\n\t"
                "vwredsum.vs v13, v28, v0\n\t"
                "vwredsum.vs v14, v30, v0\n\t"
                "vsetivli zero, 4, e32, m1\n\t"
                "vmul.vx v0, v10, t0\n\t"
                "vmul.vx v1, v9, t1\n\t"
                "vmacc.vx v0, t2, v8\n\t"
                "vmacc.vx v1, t3, v7\n\t"
                "vmacc.vx v0, t4, v11\n\t"
                "vmacc.vx v1, t5, v12\n\t"
                "vmacc.vx v0, t6, v13\n\t"
                "vmacc.vx v1, a7, v14\n\t"
                "vadd.vv v0, v0, v1\n\t"
                "vfcvt.f.x.v v0, v0\n\t"
                "vfmv.f.s %[ftmp], v0\n\t"
                "fmadd.s %[sumf], %[d], %[ftmp], %[sumf]"
                : [q6] "+&r" (q6), [q6h] "=&r" (q6h)
                , [scale] "+&r" (scale)
                , [sumf] "+&f" (sumf), [ftmp] "=&f" (ftmp)
                : [qh] "r" (qh), [q8] "r" (q8)
                , [vl32] "r" (32), [vl64] "r" (64), [vl128] "r" (128)
                , [ಮುಖವಾಡ] "r" (0x30), [d] "f" (d)
                : "memory"
                , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
                , "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a7"
                , "a6", "a5", "a4", "a3"
            );
            qh += 32;   q8 += 128;
        }
    }

    *s = sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_q6_K_q8_K_vl256(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    ದಶಮಾಂಶ sumf = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d;

        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT q6 = x[i].ql;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT qh = x[i].qh;
        ಸ್ಥಿರಾಂಕ  int8_t * GGML_RESTRICT q8 = y[i].qs;

        ಸ್ಥಿರಾಂಕ int8_t * GGML_RESTRICT scale = x[i].scales;

        size_t vl;

        vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);

        ಪೂರ್ಣಾಂಕ ಮೊತ್ತ_t = 0;
        ಪೂರ್ಣಾಂಕ is = 0;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K/128; ++j) {

            vl = 32;

            // load qh
            vuint8m1_t qh_x = __riscv_vle8_v_u8m1(qh, vl);

            // load Q6
            vuint8m1_t q6_0 = __riscv_vle8_v_u8m1(q6, vl);
            vuint8m1_t q6_1 = __riscv_vle8_v_u8m1(q6+32, vl);

            vuint8m1_t q6a_0 = __riscv_vand_vx_u8m1(q6_0, 0x0F, vl);
            vuint8m1_t q6a_1 = __riscv_vand_vx_u8m1(q6_1, 0x0F, vl);
            vuint8m1_t q6s_0 = __riscv_vsrl_vx_u8m1(q6_0, 0x04, vl);
            vuint8m1_t q6s_1 = __riscv_vsrl_vx_u8m1(q6_1, 0x04, vl);

            vuint8m1_t qh_0 = __riscv_vand_vx_u8m1(qh_x, 0x03, vl);
            vuint8m1_t qh_1 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(qh_x, 0x2, vl), 0x03 , vl);
            vuint8m1_t qh_2 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(qh_x, 0x4, vl), 0x03 , vl);
            vuint8m1_t qh_3 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(qh_x, 0x6, vl), 0x03 , vl);

            vuint8m1_t qhi_0 = __riscv_vor_vv_u8m1(q6a_0, __riscv_vsll_vx_u8m1(qh_0, 0x04, vl), vl);
            vuint8m1_t qhi_1 = __riscv_vor_vv_u8m1(q6a_1, __riscv_vsll_vx_u8m1(qh_1, 0x04, vl), vl);
            vuint8m1_t qhi_2 = __riscv_vor_vv_u8m1(q6s_0, __riscv_vsll_vx_u8m1(qh_2, 0x04, vl), vl);
            vuint8m1_t qhi_3 = __riscv_vor_vv_u8m1(q6s_1, __riscv_vsll_vx_u8m1(qh_3, 0x04, vl), vl);

            vint8m1_t a_0 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(qhi_0), 32, vl);
            vint8m1_t a_1 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(qhi_1), 32, vl);
            vint8m1_t a_2 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(qhi_2), 32, vl);
            vint8m1_t a_3 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(qhi_3), 32, vl);

            // load Q8 and take product
            vint16m2_t va_q_0 = __riscv_vwmul_vv_i16m2(a_0, __riscv_vle8_v_i8m1(q8, vl), vl);
            vint16m2_t va_q_1 = __riscv_vwmul_vv_i16m2(a_1, __riscv_vle8_v_i8m1(q8+32, vl), vl);
            vint16m2_t va_q_2 = __riscv_vwmul_vv_i16m2(a_2, __riscv_vle8_v_i8m1(q8+64, vl), vl);
            vint16m2_t va_q_3 = __riscv_vwmul_vv_i16m2(a_3, __riscv_vle8_v_i8m1(q8+96, vl), vl);

            vl = 16;

            vint32m2_t vaux_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_0, 0), scale[is+0], vl);
            vint32m2_t vaux_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_0, 1), scale[is+1], vl);
            vint32m2_t vaux_2 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_1, 0), scale[is+2], vl);
            vint32m2_t vaux_3 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_1, 1), scale[is+3], vl);
            vint32m2_t vaux_4 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_2, 0), scale[is+4], vl);
            vint32m2_t vaux_5 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_2, 1), scale[is+5], vl);
            vint32m2_t vaux_6 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_3, 0), scale[is+6], vl);
            vint32m2_t vaux_7 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_3, 1), scale[is+7], vl);

            vint32m1_t isum0 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vaux_0, vaux_1, vl), vzero, vl);
            vint32m1_t isum1 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vaux_2, vaux_3, vl), isum0, vl);
            vint32m1_t isum2 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vaux_4, vaux_5, vl), isum1, vl);
            vint32m1_t isum3 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vaux_6, vaux_7, vl), isum2, vl);

            ಮೊತ್ತ_t += __riscv_vmv_x_s_i32m1_i32(isum3);

            q6 += 64;   qh += 32;   q8 += 128;   is=8;

        }

        sumf += d * ಮೊತ್ತ_t;

    }

    *s = sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_q6_K_q8_K_vl512(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    // mask for processing 16 elements per prod register
    ಸ್ಥಿರಾಂಕ vuint16m1_t va_ಸೂಚ್ಯಂಕ = __riscv_vid_v_u16m1(32);
    ಸ್ಥಿರಾಂಕ vbool16_t va_ಮುಖವಾಡ = __riscv_vmsgtu_vx_u16m1_b16(va_ಸೂಚ್ಯಂಕ, 15, 32);

    ದಶಮಾಂಶ sumf = 0;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d;

        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT q6 = x[i].ql;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT qh = x[i].qh;
        ಸ್ಥಿರಾಂಕ  int8_t * GGML_RESTRICT q8 = y[i].qs;

        ಸ್ಥಿರಾಂಕ int8_t * GGML_RESTRICT scale = x[i].scales;

        size_t vl = 32;

        vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);

        ಪೂರ್ಣಾಂಕ ಮೊತ್ತ_t = 0;
        ಪೂರ್ಣಾಂಕ is = 0;

        vint32m2_t vaux_0 = __riscv_vmv_v_x_i32m2(0, vl);
        vint32m2_t vaux_1 = __riscv_vmv_v_x_i32m2(0, vl);
        vint32m2_t vaux_2 = __riscv_vmv_v_x_i32m2(0, vl);
        vint32m2_t vaux_3 = __riscv_vmv_v_x_i32m2(0, vl);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K/128; ++j) {
            // load qh
            vuint8mf2_t qh_x = __riscv_vle8_v_u8mf2(qh, vl);

            // load Q6
            vuint8mf2_t q6_0 = __riscv_vle8_v_u8mf2(q6, vl);
            vuint8mf2_t q6_1 = __riscv_vle8_v_u8mf2(q6+32, vl);

            vuint8mf2_t q6a_0 = __riscv_vand_vx_u8mf2(q6_0, 0x0F, vl);
            vuint8mf2_t q6a_1 = __riscv_vand_vx_u8mf2(q6_1, 0x0F, vl);
            vuint8mf2_t q6s_0 = __riscv_vsrl_vx_u8mf2(q6_0, 0x04, vl);
            vuint8mf2_t q6s_1 = __riscv_vsrl_vx_u8mf2(q6_1, 0x04, vl);

            vuint8mf2_t qh_0 = __riscv_vand_vx_u8mf2(qh_x, 0x03, vl);
            vuint8mf2_t qh_1 = __riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(qh_x, 0x2, vl), 0x03 , vl);
            vuint8mf2_t qh_2 = __riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(qh_x, 0x4, vl), 0x03 , vl);
            vuint8mf2_t qh_3 = __riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(qh_x, 0x6, vl), 0x03 , vl);

            vuint8mf2_t qhi_0 = __riscv_vor_vv_u8mf2(q6a_0, __riscv_vsll_vx_u8mf2(qh_0, 0x04, vl), vl);
            vuint8mf2_t qhi_1 = __riscv_vor_vv_u8mf2(q6a_1, __riscv_vsll_vx_u8mf2(qh_1, 0x04, vl), vl);
            vuint8mf2_t qhi_2 = __riscv_vor_vv_u8mf2(q6s_0, __riscv_vsll_vx_u8mf2(qh_2, 0x04, vl), vl);
            vuint8mf2_t qhi_3 = __riscv_vor_vv_u8mf2(q6s_1, __riscv_vsll_vx_u8mf2(qh_3, 0x04, vl), vl);

            vint8mf2_t a_0 = __riscv_vsub_vx_i8mf2(__riscv_vreinterpret_v_u8mf2_i8mf2(qhi_0), 32, vl);
            vint8mf2_t a_1 = __riscv_vsub_vx_i8mf2(__riscv_vreinterpret_v_u8mf2_i8mf2(qhi_1), 32, vl);
            vint8mf2_t a_2 = __riscv_vsub_vx_i8mf2(__riscv_vreinterpret_v_u8mf2_i8mf2(qhi_2), 32, vl);
            vint8mf2_t a_3 = __riscv_vsub_vx_i8mf2(__riscv_vreinterpret_v_u8mf2_i8mf2(qhi_3), 32, vl);

            // load Q8 and take product
            vint16m1_t va_q_0 = __riscv_vwmul_vv_i16m1(a_0, __riscv_vle8_v_i8mf2(q8, vl), vl);
            vint16m1_t va_q_1 = __riscv_vwmul_vv_i16m1(a_1, __riscv_vle8_v_i8mf2(q8+32, vl), vl);
            vint16m1_t va_q_2 = __riscv_vwmul_vv_i16m1(a_2, __riscv_vle8_v_i8mf2(q8+64, vl), vl);
            vint16m1_t va_q_3 = __riscv_vwmul_vv_i16m1(a_3, __riscv_vle8_v_i8mf2(q8+96, vl), vl);

            // accumulate
            vaux_0 = __riscv_vwmacc_vx_i32m2(vaux_0, scale[is+0], va_q_0, 16);
            vaux_1 = __riscv_vwmacc_vx_i32m2(vaux_1, scale[is+2], va_q_1, 16);
            vaux_2 = __riscv_vwmacc_vx_i32m2(vaux_2, scale[is+4], va_q_2, 16);
            vaux_3 = __riscv_vwmacc_vx_i32m2(vaux_3, scale[is+6], va_q_3, 16);
            //
            vaux_0 = __riscv_vwmacc_vx_i32m2_m(va_ಮುಖವಾಡ, vaux_0, scale[is+1], va_q_0, vl);
            vaux_1 = __riscv_vwmacc_vx_i32m2_m(va_ಮುಖವಾಡ, vaux_1, scale[is+3], va_q_1, vl);
            vaux_2 = __riscv_vwmacc_vx_i32m2_m(va_ಮುಖವಾಡ, vaux_2, scale[is+5], va_q_2, vl);
            vaux_3 = __riscv_vwmacc_vx_i32m2_m(va_ಮುಖವಾಡ, vaux_3, scale[is+7], va_q_3, vl);

            q6 += 64;   qh += 32;   q8 += 128;   is=8;
        }

        vint32m1_t isum0 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vaux_0, vaux_1, vl), vzero, vl);
        vint32m1_t isum1 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vaux_2, vaux_3, vl), isum0, vl);

        ಮೊತ್ತ_t += __riscv_vmv_x_s_i32m1_i32(isum1);

        sumf += d * ಮೊತ್ತ_t;

    }

    *s = sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_q6_K_q8_K_vl1024(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    // mask for processing 16 elements per prod register
    ಸ್ಥಿರಾಂಕ vuint16mf2_t va_ಸೂಚ್ಯಂಕ = __riscv_vid_v_u16mf2(32);
    ಸ್ಥಿರಾಂಕ vbool32_t va_ಮುಖವಾಡ = __riscv_vmsgtu_vx_u16mf2_b32(va_ಸೂಚ್ಯಂಕ, 15, 32);

    ದಶಮಾಂಶ sumf = 0;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d;

        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT q6 = x[i].ql;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT qh = x[i].qh;
        ಸ್ಥಿರಾಂಕ  int8_t * GGML_RESTRICT q8 = y[i].qs;

        ಸ್ಥಿರಾಂಕ int8_t * GGML_RESTRICT scale = x[i].scales;

        size_t vl = 32;

        vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);

        ಪೂರ್ಣಾಂಕ ಮೊತ್ತ_t = 0;
        ಪೂರ್ಣಾಂಕ is = 0;

        vint32m1_t vaux_0 = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t vaux_1 = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t vaux_2 = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t vaux_3 = __riscv_vmv_v_x_i32m1(0, vl);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K/128; ++j) {
            // load qh
            vuint8mf4_t qh_x = __riscv_vle8_v_u8mf4(qh, vl);

            // load Q6
            vuint8mf4_t q6_0 = __riscv_vle8_v_u8mf4(q6, vl);
            vuint8mf4_t q6_1 = __riscv_vle8_v_u8mf4(q6+32, vl);

            vuint8mf4_t q6a_0 = __riscv_vand_vx_u8mf4(q6_0, 0x0F, vl);
            vuint8mf4_t q6a_1 = __riscv_vand_vx_u8mf4(q6_1, 0x0F, vl);
            vuint8mf4_t q6s_0 = __riscv_vsrl_vx_u8mf4(q6_0, 0x04, vl);
            vuint8mf4_t q6s_1 = __riscv_vsrl_vx_u8mf4(q6_1, 0x04, vl);

            vuint8mf4_t qh_0 = __riscv_vand_vx_u8mf4(qh_x, 0x03, vl);
            vuint8mf4_t qh_1 = __riscv_vand_vx_u8mf4(__riscv_vsrl_vx_u8mf4(qh_x, 0x2, vl), 0x03 , vl);
            vuint8mf4_t qh_2 = __riscv_vand_vx_u8mf4(__riscv_vsrl_vx_u8mf4(qh_x, 0x4, vl), 0x03 , vl);
            vuint8mf4_t qh_3 = __riscv_vand_vx_u8mf4(__riscv_vsrl_vx_u8mf4(qh_x, 0x6, vl), 0x03 , vl);

            vuint8mf4_t qhi_0 = __riscv_vor_vv_u8mf4(q6a_0, __riscv_vsll_vx_u8mf4(qh_0, 0x04, vl), vl);
            vuint8mf4_t qhi_1 = __riscv_vor_vv_u8mf4(q6a_1, __riscv_vsll_vx_u8mf4(qh_1, 0x04, vl), vl);
            vuint8mf4_t qhi_2 = __riscv_vor_vv_u8mf4(q6s_0, __riscv_vsll_vx_u8mf4(qh_2, 0x04, vl), vl);
            vuint8mf4_t qhi_3 = __riscv_vor_vv_u8mf4(q6s_1, __riscv_vsll_vx_u8mf4(qh_3, 0x04, vl), vl);

            vint8mf4_t a_0 = __riscv_vsub_vx_i8mf4(__riscv_vreinterpret_v_u8mf4_i8mf4(qhi_0), 32, vl);
            vint8mf4_t a_1 = __riscv_vsub_vx_i8mf4(__riscv_vreinterpret_v_u8mf4_i8mf4(qhi_1), 32, vl);
            vint8mf4_t a_2 = __riscv_vsub_vx_i8mf4(__riscv_vreinterpret_v_u8mf4_i8mf4(qhi_2), 32, vl);
            vint8mf4_t a_3 = __riscv_vsub_vx_i8mf4(__riscv_vreinterpret_v_u8mf4_i8mf4(qhi_3), 32, vl);

            // load Q8 and take product
            vint16mf2_t va_q_0 = __riscv_vwmul_vv_i16mf2(a_0, __riscv_vle8_v_i8mf4(q8, vl), vl);
            vint16mf2_t va_q_1 = __riscv_vwmul_vv_i16mf2(a_1, __riscv_vle8_v_i8mf4(q8+32, vl), vl);
            vint16mf2_t va_q_2 = __riscv_vwmul_vv_i16mf2(a_2, __riscv_vle8_v_i8mf4(q8+64, vl), vl);
            vint16mf2_t va_q_3 = __riscv_vwmul_vv_i16mf2(a_3, __riscv_vle8_v_i8mf4(q8+96, vl), vl);

            // accumulate
            vaux_0 = __riscv_vwmacc_vx_i32m1(vaux_0, scale[is+0], va_q_0, 16);
            vaux_1 = __riscv_vwmacc_vx_i32m1(vaux_1, scale[is+2], va_q_1, 16);
            vaux_2 = __riscv_vwmacc_vx_i32m1(vaux_2, scale[is+4], va_q_2, 16);
            vaux_3 = __riscv_vwmacc_vx_i32m1(vaux_3, scale[is+6], va_q_3, 16);
            //
            vaux_0 = __riscv_vwmacc_vx_i32m1_m(va_ಮುಖವಾಡ, vaux_0, scale[is+1], va_q_0, vl);
            vaux_1 = __riscv_vwmacc_vx_i32m1_m(va_ಮುಖವಾಡ, vaux_1, scale[is+3], va_q_1, vl);
            vaux_2 = __riscv_vwmacc_vx_i32m1_m(va_ಮುಖವಾಡ, vaux_2, scale[is+5], va_q_2, vl);
            vaux_3 = __riscv_vwmacc_vx_i32m1_m(va_ಮುಖವಾಡ, vaux_3, scale[is+7], va_q_3, vl);

            q6 += 64;   qh += 32;   q8 += 128;   is=8;

        }

        vint32m1_t isum0 = __riscv_vredsum_vs_i32m1_i32m1(__riscv_vadd_vv_i32m1(vaux_0, vaux_1, vl), vzero, vl);
        vint32m1_t isum1 = __riscv_vredsum_vs_i32m1_i32m1(__riscv_vadd_vv_i32m1(vaux_2, vaux_3, vl), isum0, vl);

        ಮೊತ್ತ_t += __riscv_vmv_x_s_i32m1_i32(isum1);

        sumf += d * ಮೊತ್ತ_t;

    }

    *s = sumf;
}
#ಆದರೆ_ಅಂತ್ಯ

ಶೂನ್ಯ ggml_vec_dot_q6_K_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
#ಆದರೆ defined __riscv_xtheadvector
    ggml_vec_dot_q6_K_q8_K_xtheadvector(n, s, bs, vx, bx, vy, by, nrc);
#ಅಥವಾಆದರೆ defined __riscv_v
    ಆಯ್ಕೆ (__riscv_vlenb() * 8) {
        ಸ್ಥಿತಿ 128:
            ggml_vec_dot_q6_K_q8_K_vl128(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 256:
            ggml_vec_dot_q6_K_q8_K_vl256(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 512:
            ggml_vec_dot_q6_K_q8_K_vl512(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 1024:
            ggml_vec_dot_q6_K_q8_K_vl1024(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ggml_vec_dot_q6_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ggml_vec_dot_q6_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

#ಆದರೆ defined __riscv_v
ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq1_s_q8_K_vl128(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq1_s * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K  * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    ದಶಮಾಂಶ sumf = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        // Load qh once for the entire superblock.
        vuint16m1_t qh = __riscv_vle16_v_u16m1(x[i].qh, 8);

        // Calculate ls.
        vuint16m1_t ತಾತ್ಕಾಲಿಕ = __riscv_vsrl_vx_u16m1(qh, 12, 8);
        ತಾತ್ಕಾಲಿಕ = __riscv_vand_vx_u16m1(ತಾತ್ಕಾಲಿಕ, 7, 8);
        vint32m2_t ls = __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vwmulu_vx_u32m2(ತಾತ್ಕಾಲಿಕ, 2, 8));
        ls = __riscv_vadd_vx_i32m2(ls, 1, 8);

        // Calculate delta.
        vbool16_t ಮುಖವಾಡ = __riscv_vmseq_vx_u16m1_b16(__riscv_vand_vx_u16m1(qh, 0x8000, 8), 0, 8);
        vint32m2_t delta_neg = __riscv_vmv_v_x_i32m2(-1, 8);
        vint32m2_t delta_pos = __riscv_vmv_v_x_i32m2(1, 8);
        vint32m2_t delta = __riscv_vmerge_vvm_i32m2(delta_neg, delta_pos, ಮುಖವಾಡ, 8);

        // Load qs.
        vuint8m2_t qs = __riscv_vle8_v_u8m2(x[i].qs, 32);

        // Prepare the indices.
        ಸ್ಥಿರಾಂಕ uint64_t ಸರಿಸು = 0x0009000600030000;
        vuint16m4_t qh_ಸರಿಸು = __riscv_vreinterpret_v_u64m4_u16m4(__riscv_vmv_v_x_u64m4(ಸರಿಸು, 8));
        vuint16m4_t qh_gather_ಸೂಚ್ಯಂಕ = __riscv_vreinterpret_v_i16m4_u16m4(
            __riscv_vdiv_vx_i16m4(__riscv_vreinterpret_v_u16m4_i16m4(__riscv_vid_v_u16m4(32)), 4, 32));
        vuint16m4_t qh_ext = __riscv_vlmul_ext_v_u16m2_u16m4(__riscv_vlmul_ext_v_u16m1_u16m2(qh));
        vuint16m4_t qh_ಸೂಚ್ಯಂಕ = __riscv_vrgather_vv_u16m4(qh_ext, qh_gather_ಸೂಚ್ಯಂಕ, 32);
        qh_ಸೂಚ್ಯಂಕ = __riscv_vsrl_vv_u16m4(qh_ಸೂಚ್ಯಂಕ, qh_ಸರಿಸು, 32);
        qh_ಸೂಚ್ಯಂಕ = __riscv_vand_vx_u16m4(qh_ಸೂಚ್ಯಂಕ, 7, 32);
        qh_ಸೂಚ್ಯಂಕ = __riscv_vsll_vx_u16m4(qh_ಸೂಚ್ಯಂಕ, 8, 32);
        qh_ಸೂಚ್ಯಂಕ = __riscv_vor_vv_u16m4(qh_ಸೂಚ್ಯಂಕ, __riscv_vzext_vf2_u16m4(qs, 32), 32);
        vuint16m4_t ಸೂಚ್ಯಂಕ = __riscv_vsll_vx_u16m4(qh_ಸೂಚ್ಯಂಕ, 3, 32);

        // Final lsums.
        int32_t lsums_s[8];
        vint32m1_t one_scalar = __riscv_vmv_v_x_i32m1(0, 1);

        // Sub-blocks 1-2
        {
            vuint16m1_t grid_index0 = __riscv_vget_v_u16m4_u16m1(ಸೂಚ್ಯಂಕ, 0);
            vint8m4_t grid0 = __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vluxei16_v_i64m4((ಸ್ಥಿರಾಂಕ int64_t*)iq1s_grid, grid_index0, 8));
            vint8m4_t q80 = __riscv_vle8_v_i8m4(&y[i].qs[0], 64);
            vint16m8_t lsum0 = __riscv_vwmul_vv_i16m8(grid0, q80, 128);
            lsums_s[0] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(lsum0, 0), one_scalar, 32));
            lsums_s[1] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(lsum0, 1), one_scalar, 32));
        }
        __asm__ __volatile__("" ::: "memory");
        // Sub-blocks 3-4
        {
            vuint16m1_t grid_index0 = __riscv_vget_v_u16m4_u16m1(ಸೂಚ್ಯಂಕ, 1);
            vint8m4_t grid0 = __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vluxei16_v_i64m4((ಸ್ಥಿರಾಂಕ int64_t*)iq1s_grid, grid_index0, 8));
            vint8m4_t q80 = __riscv_vle8_v_i8m4(&y[i].qs[64], 64);
            vint16m8_t lsum0 = __riscv_vwmul_vv_i16m8(grid0, q80, 128);
            lsums_s[2] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(lsum0, 0), one_scalar, 32));
            lsums_s[3] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(lsum0, 1), one_scalar, 32));
        }
        __asm__ __volatile__("" ::: "memory");
        // Sub-blocks 5-6
        {
            vuint16m1_t grid_index0 = __riscv_vget_v_u16m4_u16m1(ಸೂಚ್ಯಂಕ, 2);
            vint8m4_t grid0 = __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vluxei16_v_i64m4((ಸ್ಥಿರಾಂಕ int64_t*)iq1s_grid, grid_index0, 8));
            vint8m4_t q80 = __riscv_vle8_v_i8m4(&y[i].qs[128], 64);
            vint16m8_t lsum0 = __riscv_vwmul_vv_i16m8(grid0, q80, 128);
            lsums_s[4] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(lsum0, 0), one_scalar, 32));
            lsums_s[5] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(lsum0, 1), one_scalar, 32));
        }
        __asm__ __volatile__("" ::: "memory");
        // Sub-blocks 7-8
        {
            vuint16m1_t grid_index0 = __riscv_vget_v_u16m4_u16m1(ಸೂಚ್ಯಂಕ, 3);
            vint8m4_t grid0 = __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vluxei16_v_i64m4((ಸ್ಥಿರಾಂಕ int64_t*)iq1s_grid, grid_index0, 8));
            vint8m4_t q80 = __riscv_vle8_v_i8m4(&y[i].qs[192], 64);
            vint16m8_t lsum0 = __riscv_vwmul_vv_i16m8(grid0, q80, 128);
            lsums_s[6] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(lsum0, 0), one_scalar, 32));
            lsums_s[7] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(lsum0, 1), one_scalar, 32));
        }
        __asm__ __volatile__("" ::: "memory");
        vint32m2_t lsums = __riscv_vle32_v_i32m2(&lsums_s[0], 8);

        // Calculate the bsums.
        vint16m2_t bsums_0 = __riscv_vle16_v_i16m2(y[i].bsums, 16);
        ಸ್ಥಿರಾಂಕ vuint32m2_t bsums_i32 = __riscv_vreinterpret_v_u16m2_u32m2(__riscv_vreinterpret_v_i16m2_u16m2(bsums_0));
        ಸ್ಥಿರಾಂಕ vint16m1_t bsums_i32_0 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vnsrl_wx_u16m1(bsums_i32, 0, 8));
        ಸ್ಥಿರಾಂಕ vint16m1_t bsums_i32_1 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vnsrl_wx_u16m1(bsums_i32, 16, 8));
        ಸ್ಥಿರಾಂಕ vint32m2_t bsums = __riscv_vwadd_vv_i32m2(bsums_i32_0, bsums_i32_1, 8);

        // Accumulation.
        vint32m2_t sumi_v = __riscv_vmul_vv_i32m2(ls, lsums, 8);
        vint32m2_t sumi1_v = __riscv_vmul_vv_i32m2(__riscv_vmul_vv_i32m2(ls, delta, 8), bsums, 8);

        // Update sumf.
        ಪೂರ್ಣಾಂಕ sumi = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m2_i32m1(sumi_v, __riscv_vmv_v_x_i32m1(0.0f, 1), 8));
        ಪೂರ್ಣಾಂಕ sumi1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m2_i32m1(sumi1_v, __riscv_vmv_v_x_i32m1(0.0f, 1), 8));
        sumf += GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d * (sumi + IQ1S_DELTA * sumi1);
    }

    *s = sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq1_s_q8_K_vl256(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq1_s * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K  * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    ದಶಮಾಂಶ sumf = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        // Load qh once for the entire superblock.
        vuint16mf2_t qh = __riscv_vle16_v_u16mf2(x[i].qh, 8);

        // Calculate ls.
        vuint16mf2_t ತಾತ್ಕಾಲಿಕ = __riscv_vsrl_vx_u16mf2(qh, 12, 8);
        ತಾತ್ಕಾಲಿಕ = __riscv_vand_vx_u16mf2(ತಾತ್ಕಾಲಿಕ, 7, 8);
        vint32m1_t ls = __riscv_vreinterpret_v_u32m1_i32m1(__riscv_vwmulu_vx_u32m1(ತಾತ್ಕಾಲಿಕ, 2, 8));
        ls = __riscv_vadd_vx_i32m1(ls, 1, 8);

        // Calculate delta.
        vbool32_t ಮುಖವಾಡ = __riscv_vmseq_vx_u16mf2_b32(__riscv_vand_vx_u16mf2(qh, 0x8000, 8), 0, 8);
        vint32m1_t delta_neg = __riscv_vmv_v_x_i32m1(-1, 8);
        vint32m1_t delta_pos = __riscv_vmv_v_x_i32m1(1, 8);
        vint32m1_t delta = __riscv_vmerge_vvm_i32m1(delta_neg, delta_pos, ಮುಖವಾಡ, 8);

        // Load qs.
        vuint8m1_t qs = __riscv_vle8_v_u8m1(x[i].qs, 32);

        // Prepare the indices.
        ಸ್ಥಿರಾಂಕ uint64_t ಸರಿಸು = 0x0009000600030000;
        vuint16m2_t qh_ಸರಿಸು = __riscv_vreinterpret_v_u64m2_u16m2(__riscv_vmv_v_x_u64m2(ಸರಿಸು, 8));
        vuint16m2_t qh_gather_ಸೂಚ್ಯಂಕ = __riscv_vreinterpret_v_i16m2_u16m2(
            __riscv_vdiv_vx_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vid_v_u16m2(32)), 4, 32));
        vuint16m2_t qh_ext = __riscv_vlmul_ext_v_u16m1_u16m2(__riscv_vlmul_ext_v_u16mf2_u16m1(qh));
        vuint16m2_t qh_ಸೂಚ್ಯಂಕ = __riscv_vrgather_vv_u16m2(qh_ext, qh_gather_ಸೂಚ್ಯಂಕ, 32);
        qh_ಸೂಚ್ಯಂಕ = __riscv_vsrl_vv_u16m2(qh_ಸೂಚ್ಯಂಕ, qh_ಸರಿಸು, 32);
        qh_ಸೂಚ್ಯಂಕ = __riscv_vand_vx_u16m2(qh_ಸೂಚ್ಯಂಕ, 7, 32);
        qh_ಸೂಚ್ಯಂಕ = __riscv_vsll_vx_u16m2(qh_ಸೂಚ್ಯಂಕ, 8, 32);
        qh_ಸೂಚ್ಯಂಕ = __riscv_vor_vv_u16m2(qh_ಸೂಚ್ಯಂಕ, __riscv_vzext_vf2_u16m2(qs, 32), 32);
        vuint16m2_t ಸೂಚ್ಯಂಕ = __riscv_vsll_vx_u16m2(qh_ಸೂಚ್ಯಂಕ, 3, 32);

        // Final lsums.
        int32_t lsums_s[8];
        vint32m1_t one_scalar = __riscv_vmv_v_x_i32m1(0, 1);

        // Sub-blocks 1-4
        {
            vuint16m1_t grid_index0 = __riscv_vget_v_u16m2_u16m1(ಸೂಚ್ಯಂಕ, 0);
            vint8m4_t grid0 = __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vluxei16_v_i64m4((ಸ್ಥಿರಾಂಕ int64_t*)iq1s_grid, grid_index0, 16));
            vint8m4_t q80 = __riscv_vle8_v_i8m4(y[i].qs, 128);
            vint16m8_t lsum0 = __riscv_vwmul_vv_i16m8(grid0, q80, 128);
            lsums_s[0] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(lsum0, 0), one_scalar, 32));
            lsums_s[1] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(lsum0, 1), one_scalar, 32));
            lsums_s[2] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(lsum0, 2), one_scalar, 32));
            lsums_s[3] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(lsum0, 3), one_scalar, 32));
        }
        __asm__ __volatile__("" ::: "memory");
        // Sub-blocks 5-8
        {
            vuint16m1_t grid_index1 = __riscv_vget_v_u16m2_u16m1(ಸೂಚ್ಯಂಕ, 1);
            vint8m4_t grid1 = __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vluxei16_v_i64m4((ಸ್ಥಿರಾಂಕ int64_t*)iq1s_grid, grid_index1, 16));
            vint8m4_t q81 = __riscv_vle8_v_i8m4(&y[i].qs[128], 128);
            vint16m8_t lsum1 = __riscv_vwmul_vv_i16m8(grid1, q81, 128);
            lsums_s[4] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(lsum1, 0), one_scalar, 32));
            lsums_s[5] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(lsum1, 1), one_scalar, 32));
            lsums_s[6] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(lsum1, 2), one_scalar, 32));
            lsums_s[7] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(lsum1, 3), one_scalar, 32));
        }
        __asm__ __volatile__("" ::: "memory");
        vint32m1_t lsums = __riscv_vle32_v_i32m1(&lsums_s[0], 8);

        // Calculate the bsums.
        vint16m1_t bsums_0 = __riscv_vle16_v_i16m1(y[i].bsums, 16);
        ಸ್ಥಿರಾಂಕ vuint32m1_t bsums_i32 = __riscv_vreinterpret_v_u16m1_u32m1(__riscv_vreinterpret_v_i16m1_u16m1(bsums_0));
        ಸ್ಥಿರಾಂಕ vint16mf2_t bsums_i32_0 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(bsums_i32, 0, 8));
        ಸ್ಥಿರಾಂಕ vint16mf2_t bsums_i32_1 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(bsums_i32, 16, 8));
        ಸ್ಥಿರಾಂಕ vint32m1_t bsums = __riscv_vwadd_vv_i32m1(bsums_i32_0, bsums_i32_1, 8);

        // Accumulation.
        vint32m1_t sumi_v = __riscv_vmul_vv_i32m1(ls, lsums, 8);
        vint32m1_t sumi1_v = __riscv_vmul_vv_i32m1(__riscv_vmul_vv_i32m1(ls, delta, 8), bsums, 8);

        // Update sumf.
        ಪೂರ್ಣಾಂಕ sumi = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m1_i32m1(sumi_v, __riscv_vmv_v_x_i32m1(0.0f, 1), 8));
        ಪೂರ್ಣಾಂಕ sumi1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m1_i32m1(sumi1_v, __riscv_vmv_v_x_i32m1(0.0f, 1), 8));
        sumf += GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d * (sumi + IQ1S_DELTA * sumi1);
    }

    *s = sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq1_s_q8_K_vl512(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq1_s * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K  * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    ದಶಮಾಂಶ sumf = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        // Load qh once for the entire superblock.
        vuint16mf4_t qh = __riscv_vle16_v_u16mf4(x[i].qh, 8);

        // Calculate ls.
        vuint16mf4_t ತಾತ್ಕಾಲಿಕ = __riscv_vsrl_vx_u16mf4(qh, 12, 8);
        ತಾತ್ಕಾಲಿಕ = __riscv_vand_vx_u16mf4(ತಾತ್ಕಾಲಿಕ, 7, 8);
        vint32mf2_t ls = __riscv_vreinterpret_v_u32mf2_i32mf2(__riscv_vwmulu_vx_u32mf2(ತಾತ್ಕಾಲಿಕ, 2, 8));
        ls = __riscv_vadd_vx_i32mf2(ls, 1, 8);

        // Calculate delta.
        vbool64_t ಮುಖವಾಡ = __riscv_vmseq_vx_u16mf4_b64(__riscv_vand_vx_u16mf4(qh, 0x8000, 8), 0, 8);
        vint32mf2_t delta_neg = __riscv_vmv_v_x_i32mf2(-1, 8);
        vint32mf2_t delta_pos = __riscv_vmv_v_x_i32mf2(1, 8);
        vint32mf2_t delta = __riscv_vmerge_vvm_i32mf2(delta_neg, delta_pos, ಮುಖವಾಡ, 8);

        // Load qs.
        vuint8mf2_t qs = __riscv_vle8_v_u8mf2(x[i].qs, 32);

        // Prepare the indices.
        ಸ್ಥಿರಾಂಕ uint64_t ಸರಿಸು = 0x0009000600030000;
        vuint16m1_t qh_ಸರಿಸು = __riscv_vreinterpret_v_u64m1_u16m1(__riscv_vmv_v_x_u64m1(ಸರಿಸು, 8));
        vuint16m1_t qh_gather_ಸೂಚ್ಯಂಕ = __riscv_vreinterpret_v_i16m1_u16m1(
            __riscv_vdiv_vx_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vid_v_u16m1(32)), 4, 32));
        vuint16m1_t qh_ext = __riscv_vlmul_ext_v_u16mf2_u16m1(__riscv_vlmul_ext_v_u16mf4_u16mf2(qh));
        vuint16m1_t qh_ಸೂಚ್ಯಂಕ = __riscv_vrgather_vv_u16m1(qh_ext, qh_gather_ಸೂಚ್ಯಂಕ, 32);
        qh_ಸೂಚ್ಯಂಕ = __riscv_vsrl_vv_u16m1(qh_ಸೂಚ್ಯಂಕ, qh_ಸರಿಸು, 32);
        qh_ಸೂಚ್ಯಂಕ = __riscv_vand_vx_u16m1(qh_ಸೂಚ್ಯಂಕ, 7, 32);
        qh_ಸೂಚ್ಯಂಕ = __riscv_vsll_vx_u16m1(qh_ಸೂಚ್ಯಂಕ, 8, 32);
        qh_ಸೂಚ್ಯಂಕ = __riscv_vor_vv_u16m1(qh_ಸೂಚ್ಯಂಕ, __riscv_vzext_vf2_u16m1(qs, 32), 32);
        vuint16m1_t ಸೂಚ್ಯಂಕ = __riscv_vsll_vx_u16m1(qh_ಸೂಚ್ಯಂಕ, 3, 32);

        // Final lsums.
        int32_t lsums_s[8];
        vint32m1_t one_scalar = __riscv_vmv_v_x_i32m1(0, 1);

        // Sub-blocks 1-8
        {
            vint8m4_t grid0 = __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vluxei16_v_i64m4((ಸ್ಥಿರಾಂಕ int64_t*)iq1s_grid, ಸೂಚ್ಯಂಕ, 32));
            vint8m4_t q80 = __riscv_vle8_v_i8m4(y[i].qs, 256);
            vint16m8_t lsum0 = __riscv_vwmul_vv_i16m8(grid0, q80, 256);
            lsums_s[0] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(lsum0, 0), one_scalar, 32));
            lsums_s[1] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(lsum0, 1), one_scalar, 32));
            lsums_s[2] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(lsum0, 2), one_scalar, 32));
            lsums_s[3] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(lsum0, 3), one_scalar, 32));
            lsums_s[4] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(lsum0, 4), one_scalar, 32));
            lsums_s[5] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(lsum0, 5), one_scalar, 32));
            lsums_s[6] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(lsum0, 6), one_scalar, 32));
            lsums_s[7] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(lsum0, 7), one_scalar, 32));
        }
        __asm__ __volatile__("" ::: "memory");
        vint32mf2_t lsums = __riscv_vle32_v_i32mf2(&lsums_s[0], 8);

        // Calculate the bsums.
        vint16mf2_t bsums_0 = __riscv_vle16_v_i16mf2(y[i].bsums, 16);
        ಸ್ಥಿರಾಂಕ vuint32mf2_t bsums_i32 = __riscv_vreinterpret_v_u16mf2_u32mf2(__riscv_vreinterpret_v_i16mf2_u16mf2(bsums_0));
        ಸ್ಥಿರಾಂಕ vint16mf4_t bsums_i32_0 = __riscv_vreinterpret_v_u16mf4_i16mf4(__riscv_vnsrl_wx_u16mf4(bsums_i32, 0, 8));
        ಸ್ಥಿರಾಂಕ vint16mf4_t bsums_i32_1 = __riscv_vreinterpret_v_u16mf4_i16mf4(__riscv_vnsrl_wx_u16mf4(bsums_i32, 16, 8));
        ಸ್ಥಿರಾಂಕ vint32mf2_t bsums = __riscv_vwadd_vv_i32mf2(bsums_i32_0, bsums_i32_1, 8);

        // Accumulation.
        vint32mf2_t sumi_v = __riscv_vmul_vv_i32mf2(ls, lsums, 8);
        vint32mf2_t sumi1_v = __riscv_vmul_vv_i32mf2(__riscv_vmul_vv_i32mf2(ls, delta, 8), bsums, 8);

        // Update sumf.
        ಪೂರ್ಣಾಂಕ sumi = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32mf2_i32m1(sumi_v, __riscv_vmv_v_x_i32m1(0.0f, 1), 8));
        ಪೂರ್ಣಾಂಕ sumi1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32mf2_i32m1(sumi1_v, __riscv_vmv_v_x_i32m1(0.0f, 1), 8));
        sumf += GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d * (sumi + IQ1S_DELTA * sumi1);
    }

    *s = sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq1_s_q8_K_vl1024(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq1_s * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K  * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    // Mask for processing 32 elements per lsum register.
    vuint16m1_t l_ಸೂಚ್ಯಂಕ = __riscv_vid_v_u16m1(64);
    vbool16_t l_ಮುಖವಾಡ = __riscv_vmsgtu_vx_u16m1_b16(l_ಸೂಚ್ಯಂಕ, 31, 64);

    ದಶಮಾಂಶ sumf = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        // Load qh once for the entire superblock.
        vuint16mf4_t qh = __riscv_vle16_v_u16mf4(x[i].qh, 8);

        // Calculate ls.
        vuint16mf4_t ತಾತ್ಕಾಲಿಕ = __riscv_vsrl_vx_u16mf4(qh, 12, 8);
        ತಾತ್ಕಾಲಿಕ = __riscv_vand_vx_u16mf4(ತಾತ್ಕಾಲಿಕ, 7, 8);
        vint32mf2_t ls = __riscv_vreinterpret_v_u32mf2_i32mf2(__riscv_vwmulu_vx_u32mf2(ತಾತ್ಕಾಲಿಕ, 2, 8));
        ls = __riscv_vadd_vx_i32mf2(ls, 1, 8);

        // Calculate delta.
        vbool64_t ಮುಖವಾಡ = __riscv_vmseq_vx_u16mf4_b64(__riscv_vand_vx_u16mf4(qh, 0x8000, 8), 0, 8);
        vint32mf2_t delta_neg = __riscv_vmv_v_x_i32mf2(-1, 8);
        vint32mf2_t delta_pos = __riscv_vmv_v_x_i32mf2(1, 8);
        vint32mf2_t delta = __riscv_vmerge_vvm_i32mf2(delta_neg, delta_pos, ಮುಖವಾಡ, 8);

        // Load qs.
        vuint8mf2_t qs = __riscv_vle8_v_u8mf2(x[i].qs, 32);

        // Prepare the indices.
        ಸ್ಥಿರಾಂಕ uint64_t ಸರಿಸು = 0x0009000600030000;
        vuint16m1_t qh_ಸರಿಸು = __riscv_vreinterpret_v_u64m1_u16m1(__riscv_vmv_v_x_u64m1(ಸರಿಸು, 8));
        vuint16m1_t qh_gather_ಸೂಚ್ಯಂಕ = __riscv_vreinterpret_v_i16m1_u16m1(
            __riscv_vdiv_vx_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vid_v_u16m1(32)), 4, 32));
        vuint16m1_t qh_ext = __riscv_vlmul_ext_v_u16mf2_u16m1(__riscv_vlmul_ext_v_u16mf4_u16mf2(qh));
        vuint16m1_t qh_ಸೂಚ್ಯಂಕ = __riscv_vrgather_vv_u16m1(qh_ext, qh_gather_ಸೂಚ್ಯಂಕ, 32);
        qh_ಸೂಚ್ಯಂಕ = __riscv_vsrl_vv_u16m1(qh_ಸೂಚ್ಯಂಕ, qh_ಸರಿಸು, 32);
        qh_ಸೂಚ್ಯಂಕ = __riscv_vand_vx_u16m1(qh_ಸೂಚ್ಯಂಕ, 7, 32);
        qh_ಸೂಚ್ಯಂಕ = __riscv_vsll_vx_u16m1(qh_ಸೂಚ್ಯಂಕ, 8, 32);
        qh_ಸೂಚ್ಯಂಕ = __riscv_vor_vv_u16m1(qh_ಸೂಚ್ಯಂಕ, __riscv_vzext_vf2_u16m1(qs, 32), 32);
        vuint16mf2_t ಸೂಚ್ಯಂಕ = __riscv_vlmul_trunc_v_u16m1_u16mf2(__riscv_vsll_vx_u16m1(qh_ಸೂಚ್ಯಂಕ, 3, 32));

        // Final lsums.
        int32_t lsums_s[8];
        vint32m1_t one_scalar = __riscv_vmv_v_x_i32m1(0, 1);

        // Sub-blocks 1-8
        {
            vint8m2_t grid0 = __riscv_vreinterpret_v_i64m2_i8m2(__riscv_vluxei16_v_i64m2((ಸ್ಥಿರಾಂಕ int64_t*)iq1s_grid, ಸೂಚ್ಯಂಕ, 32));
            vint8m2_t q80 = __riscv_vle8_v_i8m2(y[i].qs, 256);
            vint16m4_t lsum0 = __riscv_vwmul_vv_i16m4(grid0, q80, 256);

            // Reduce.
            lsums_s[0] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(          __riscv_vget_v_i16m4_i16m1(lsum0, 0), one_scalar, 32));
            lsums_s[1] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1_m(l_ಮುಖವಾಡ, __riscv_vget_v_i16m4_i16m1(lsum0, 0), one_scalar, 64));
            lsums_s[2] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(          __riscv_vget_v_i16m4_i16m1(lsum0, 1), one_scalar, 32));
            lsums_s[3] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1_m(l_ಮುಖವಾಡ, __riscv_vget_v_i16m4_i16m1(lsum0, 1), one_scalar, 64));
            lsums_s[4] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(          __riscv_vget_v_i16m4_i16m1(lsum0, 2), one_scalar, 32));
            lsums_s[5] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1_m(l_ಮುಖವಾಡ, __riscv_vget_v_i16m4_i16m1(lsum0, 2), one_scalar, 64));
            lsums_s[6] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(          __riscv_vget_v_i16m4_i16m1(lsum0, 3), one_scalar, 32));
            lsums_s[7] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1_m(l_ಮುಖವಾಡ, __riscv_vget_v_i16m4_i16m1(lsum0, 3), one_scalar, 64));
        }
        __asm__ __volatile__("" ::: "memory");
        vint32mf2_t lsums = __riscv_vle32_v_i32mf2(&lsums_s[0], 8);

        // Calculate the bsums.
        vint16mf2_t bsums_0 = __riscv_vle16_v_i16mf2(y[i].bsums, 16);
        ಸ್ಥಿರಾಂಕ vuint32mf2_t bsums_i32 = __riscv_vreinterpret_v_u16mf2_u32mf2(__riscv_vreinterpret_v_i16mf2_u16mf2(bsums_0));
        ಸ್ಥಿರಾಂಕ vint16mf4_t bsums_i32_0 = __riscv_vreinterpret_v_u16mf4_i16mf4(__riscv_vnsrl_wx_u16mf4(bsums_i32, 0, 8));
        ಸ್ಥಿರಾಂಕ vint16mf4_t bsums_i32_1 = __riscv_vreinterpret_v_u16mf4_i16mf4(__riscv_vnsrl_wx_u16mf4(bsums_i32, 16, 8));
        ಸ್ಥಿರಾಂಕ vint32mf2_t bsums = __riscv_vwadd_vv_i32mf2(bsums_i32_0, bsums_i32_1, 8);

        // Accumulation.
        vint32mf2_t sumi_v = __riscv_vmul_vv_i32mf2(ls, lsums, 8);
        vint32mf2_t sumi1_v = __riscv_vmul_vv_i32mf2(__riscv_vmul_vv_i32mf2(ls, delta, 8), bsums, 8);

        // Update sumf.
        ಪೂರ್ಣಾಂಕ sumi = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32mf2_i32m1(sumi_v, __riscv_vmv_v_x_i32m1(0.0f, 1), 8));
        ಪೂರ್ಣಾಂಕ sumi1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32mf2_i32m1(sumi1_v, __riscv_vmv_v_x_i32m1(0.0f, 1), 8));
        sumf += GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d * (sumi + IQ1S_DELTA * sumi1);
    }

    *s = sumf;
}
#ಆದರೆ_ಅಂತ್ಯ

ಶೂನ್ಯ ggml_vec_dot_iq1_s_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
#ಆದರೆ defined __riscv_v
    ಆಯ್ಕೆ (__riscv_vlenb() * 8) {
        ಸ್ಥಿತಿ 128:
            ggml_vec_dot_iq1_s_q8_K_vl128(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 256:
            ggml_vec_dot_iq1_s_q8_K_vl256(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 512:
            ggml_vec_dot_iq1_s_q8_K_vl512(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 1024:
            ggml_vec_dot_iq1_s_q8_K_vl1024(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ggml_vec_dot_iq1_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ggml_vec_dot_iq1_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

#ಆದರೆ defined __riscv_v
ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq1_m_q8_K_vl128(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq1_m * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K  * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    iq1m_scale_t scale;
    ದಶಮಾಂಶ sumf = 0.0f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ int8_t   * q8 = y[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t  * qs = x[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t  * qh = x[i].qh;
        ಸ್ಥಿರಾಂಕ uint16_t * sc = (ಸ್ಥಿರಾಂಕ uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        // Accumulators.
        vint32m4_t acc1 = __riscv_vmv_v_x_i32m4(0, 16);
        vint32m4_t acc2 = __riscv_vmv_v_x_i32m4(0, 16);

        // We process 8 16-element sub-blocks together.
        #ಸೂಚನೆ GCC unroll 1
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ib = 0; ib < QK_K/128; ib++) {
            // Load qh for 8 sub-blocks.
            ಸ್ಥಿರಾಂಕ vuint8mf2_t qh_8 = __riscv_vle8_v_u8mf2(qh, 8);
            ಸ್ಥಿರಾಂಕ vuint16m1_t qh_16_lo = __riscv_vzext_vf2_u16m1(qh_8, 8);
            ಸ್ಥಿರಾಂಕ vuint16m1_t qh_16_hi = __riscv_vsll_vx_u16m1(qh_16_lo, 8, 8);
            ಸ್ಥಿರಾಂಕ vuint16m2_t qhb = __riscv_vzext_vf2_u16m2(
                __riscv_vreinterpret_v_u16m1_u8m1(__riscv_vor_vv_u16m1(qh_16_lo, qh_16_hi, 8)), 16);
            qh += 8;

            // Prepare grid indices.
            ಸ್ಥಿರಾಂಕ vuint16m2_t qsb = __riscv_vzext_vf2_u16m2(__riscv_vle8_v_u8m1(&qs[0], 16), 16);
            ಸ್ಥಿರಾಂಕ vuint16m2_t ಸರಿಸು = __riscv_vreinterpret_v_u32m2_u16m2(__riscv_vmv_v_x_u32m2(0x00040008, 8));
            vuint16m2_t ಸೂಚ್ಯಂಕ = __riscv_vor_vv_u16m2(qsb, __riscv_vand_vx_u16m2(__riscv_vsll_vv_u16m2(qhb, ಸರಿಸು, 16), 0x700, 16), 16);
            ಸೂಚ್ಯಂಕ = __riscv_vsll_vx_u16m2(ಸೂಚ್ಯಂಕ, 3, 16);
            qs += 16;

            // Prepare the deltas.
            ಸ್ಥಿರಾಂಕ vbool8_t ಮುಖವಾಡ = __riscv_vmsgtu_vx_u16m2_b8(
                __riscv_vand_vv_u16m2(qhb, __riscv_vreinterpret_v_u32m2_u16m2(__riscv_vmv_v_x_u32m2(0x00800008, 8)), 16), 0, 16);
            ಸ್ಥಿರಾಂಕ vint64m8_t delta_pos = __riscv_vmv_v_x_i64m8(0x0101010101010101, 16);
            ಸ್ಥಿರಾಂಕ vint8m8_t delta = __riscv_vreinterpret_v_i64m8_i8m8(
                __riscv_vmerge_vxm_i64m8(delta_pos, 0xffffffffffffffff, ಮುಖವಾಡ, 16));

            // Sub-blocks 0-3
            {
                // Load the grid.
                ಸ್ಥಿರಾಂಕ vint8m4_t iq1b = __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vreinterpret_v_u64m4_i64m4(
                    __riscv_vluxei16_v_u64m4(iq1s_grid, __riscv_vget_v_u16m2_u16m1(ಸೂಚ್ಯಂಕ, 0), 8)));

                // Calculate the lsums.
                //
                // Sub-block 0, 1
                {
                    // Load q8 for each sub-block.
                    ಸ್ಥಿರಾಂಕ vint8m2_t q8b = __riscv_vle8_v_i8m2(q8, 32);
                    q8 += 32;

                    // Calculate the lsums.
                    ಸ್ಥಿರಾಂಕ vint16m4_t lsum1 = __riscv_vwmul_vv_i16m4(__riscv_vget_v_i8m4_i8m2(iq1b, 0), q8b, 32);
                    ಸ್ಥಿರಾಂಕ vint16m4_t lsum2 = __riscv_vwmul_vv_i16m4(__riscv_vget_v_i8m8_i8m2(delta, 0), q8b, 32);

                    // Prepare the scales.
                    ಸ್ಥಿರಾಂಕ int16_t ls_0 = 2*((sc[0] >> 0) & 0x7) + 1;
                    ಸ್ಥಿರಾಂಕ int16_t ls_1 = 2*((sc[0] >> 3) & 0x7) + 1;

                    // Accumulate in acc0 and acc1 for each sub-block.
                    acc1 = __riscv_vwmacc_vx_i32m4(acc1, ls_0, __riscv_vget_v_i16m4_i16m2(lsum1, 0), 16);
                    acc1 = __riscv_vwmacc_vx_i32m4(acc1, ls_1, __riscv_vget_v_i16m4_i16m2(lsum1, 1), 16);
                    acc2 = __riscv_vwmacc_vx_i32m4(acc2, ls_0, __riscv_vget_v_i16m4_i16m2(lsum2, 0), 16);
                    acc2 = __riscv_vwmacc_vx_i32m4(acc2, ls_1, __riscv_vget_v_i16m4_i16m2(lsum2, 1), 16);
                }
                __asm__ __volatile__("" ::: "memory");
                // Sub-block 2, 3
                {
                    // Load q8 for each sub-block.
                    ಸ್ಥಿರಾಂಕ vint8m2_t q8b = __riscv_vle8_v_i8m2(q8, 32);
                    q8 += 32;

                    // Calculate the lsums.
                    ಸ್ಥಿರಾಂಕ vint16m4_t lsum1 = __riscv_vwmul_vv_i16m4(__riscv_vget_v_i8m4_i8m2(iq1b, 1), q8b, 32);
                    ಸ್ಥಿರಾಂಕ vint16m4_t lsum2 = __riscv_vwmul_vv_i16m4(__riscv_vget_v_i8m8_i8m2(delta, 1), q8b, 32);

                    // Prepare the scales.
                    ಸ್ಥಿರಾಂಕ int16_t ls_0 = 2*((sc[0] >> 6) & 0x7) + 1;
                    ಸ್ಥಿರಾಂಕ int16_t ls_1 = 2*((sc[0] >> 9) & 0x7) + 1;

                    // Accumulate in acc0 and acc1 for each sub-block.
                    acc1 = __riscv_vwmacc_vx_i32m4(acc1, ls_0, __riscv_vget_v_i16m4_i16m2(lsum1, 0), 16);
                    acc1 = __riscv_vwmacc_vx_i32m4(acc1, ls_1, __riscv_vget_v_i16m4_i16m2(lsum1, 1), 16);
                    acc2 = __riscv_vwmacc_vx_i32m4(acc2, ls_0, __riscv_vget_v_i16m4_i16m2(lsum2, 0), 16);
                    acc2 = __riscv_vwmacc_vx_i32m4(acc2, ls_1, __riscv_vget_v_i16m4_i16m2(lsum2, 1), 16);
                }
                sc += 1;
            }
            __asm__ __volatile__("" ::: "memory");
            // Sub-blocks 4-7
            {
                // Load the grid.
                ಸ್ಥಿರಾಂಕ vint8m4_t iq1b = __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vreinterpret_v_u64m4_i64m4(
                    __riscv_vluxei16_v_u64m4(iq1s_grid, __riscv_vget_v_u16m2_u16m1(ಸೂಚ್ಯಂಕ, 1), 8)));

                // Calculate the lsums.
                //
                // Sub-block 4, 5
                {
                    // Load q8 for each sub-block.
                    ಸ್ಥಿರಾಂಕ vint8m2_t q8b = __riscv_vle8_v_i8m2(q8, 32);
                    q8 += 32;

                    // Calculate the lsums.
                    ಸ್ಥಿರಾಂಕ vint16m4_t lsum1 = __riscv_vwmul_vv_i16m4(__riscv_vget_v_i8m4_i8m2(iq1b, 0), q8b, 32);
                    ಸ್ಥಿರಾಂಕ vint16m4_t lsum2 = __riscv_vwmul_vv_i16m4(__riscv_vget_v_i8m8_i8m2(delta, 2), q8b, 32);

                    // Prepare the scales.
                    ಸ್ಥಿರಾಂಕ int16_t ls_0 = 2*((sc[0] >> 0) & 0x7) + 1;
                    ಸ್ಥಿರಾಂಕ int16_t ls_1 = 2*((sc[0] >> 3) & 0x7) + 1;

                    // Accumulate in acc0 and acc1 for each sub-block.
                    acc1 = __riscv_vwmacc_vx_i32m4(acc1, ls_0, __riscv_vget_v_i16m4_i16m2(lsum1, 0), 16);
                    acc1 = __riscv_vwmacc_vx_i32m4(acc1, ls_1, __riscv_vget_v_i16m4_i16m2(lsum1, 1), 16);
                    acc2 = __riscv_vwmacc_vx_i32m4(acc2, ls_0, __riscv_vget_v_i16m4_i16m2(lsum2, 0), 16);
                    acc2 = __riscv_vwmacc_vx_i32m4(acc2, ls_1, __riscv_vget_v_i16m4_i16m2(lsum2, 1), 16);
                }
                __asm__ __volatile__("" ::: "memory");
                // Sub-block 6, 7
                {
                    // Load q8 for each sub-block.
                    ಸ್ಥಿರಾಂಕ vint8m2_t q8b = __riscv_vle8_v_i8m2(q8, 32);
                    q8 += 32;

                    // Calculate the lsums.
                    ಸ್ಥಿರಾಂಕ vint16m4_t lsum1 = __riscv_vwmul_vv_i16m4(__riscv_vget_v_i8m4_i8m2(iq1b, 1), q8b, 32);
                    ಸ್ಥಿರಾಂಕ vint16m4_t lsum2 = __riscv_vwmul_vv_i16m4(__riscv_vget_v_i8m8_i8m2(delta, 3), q8b, 32);

                    // Prepare the scales.
                    ಸ್ಥಿರಾಂಕ int16_t ls_0 = 2*((sc[0] >> 6) & 0x7) + 1;
                    ಸ್ಥಿರಾಂಕ int16_t ls_1 = 2*((sc[0] >> 9) & 0x7) + 1;

                    // Accumulate in acc0 and acc1 for each sub-block.
                    acc1 = __riscv_vwmacc_vx_i32m4(acc1, ls_0, __riscv_vget_v_i16m4_i16m2(lsum1, 0), 16);
                    acc1 = __riscv_vwmacc_vx_i32m4(acc1, ls_1, __riscv_vget_v_i16m4_i16m2(lsum1, 1), 16);
                    acc2 = __riscv_vwmacc_vx_i32m4(acc2, ls_0, __riscv_vget_v_i16m4_i16m2(lsum2, 0), 16);
                    acc2 = __riscv_vwmacc_vx_i32m4(acc2, ls_1, __riscv_vget_v_i16m4_i16m2(lsum2, 1), 16);
                }
                sc += 1;
            }
        }

        // Reduce and accumulate in `sumf`.
        vint32m1_t one = __riscv_vmv_v_x_i32m1(0, 1);
        ಪೂರ್ಣಾಂಕ sumi1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(acc1, one, 16));
        ಪೂರ್ಣಾಂಕ sumi2 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(acc2, one, 16));
        sumf += y[i].d * GGML_CPU_FP16_ಗಮ್ಯ_FP32(scale.f16) * (sumi1 + IQ1M_DELTA * sumi2);
    }

    *s = sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq1_m_q8_K_vl256(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq1_m * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K  * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    iq1m_scale_t scale;
    ದಶಮಾಂಶ sumf = 0.0f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ int8_t   * q8 = y[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t  * qs = x[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t  * qh = x[i].qh;
        ಸ್ಥಿರಾಂಕ uint16_t * sc = (ಸ್ಥಿರಾಂಕ uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        // Accumulators.
        vint32m2_t acc1 = __riscv_vmv_v_x_i32m2(0, 16);
        vint32m2_t acc2 = __riscv_vmv_v_x_i32m2(0, 16);

        // We process 8 16-element sub-blocks together.
        #ಸೂಚನೆ GCC unroll 1
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ib = 0; ib < QK_K/128; ib++) {
            // Load qh for 8 sub-blocks.
            ಸ್ಥಿರಾಂಕ vuint8mf4_t qh_8 = __riscv_vle8_v_u8mf4(qh, 8);
            ಸ್ಥಿರಾಂಕ vuint16mf2_t qh_16_lo = __riscv_vzext_vf2_u16mf2(qh_8, 8);
            ಸ್ಥಿರಾಂಕ vuint16mf2_t qh_16_hi = __riscv_vsll_vx_u16mf2(qh_16_lo, 8, 8);
            ಸ್ಥಿರಾಂಕ vuint16m1_t qhb = __riscv_vzext_vf2_u16m1(
                __riscv_vreinterpret_v_u16mf2_u8mf2(__riscv_vor_vv_u16mf2(qh_16_lo, qh_16_hi, 8)), 16);
            qh += 8;

            __asm__ __volatile__("" ::: "memory");

            // Prepare grid indices.
            ಸ್ಥಿರಾಂಕ vuint16m1_t qsb = __riscv_vzext_vf2_u16m1(__riscv_vle8_v_u8mf2(&qs[0], 16), 16);
            ಸ್ಥಿರಾಂಕ vuint16m1_t ಸರಿಸು = __riscv_vreinterpret_v_u32m1_u16m1(__riscv_vmv_v_x_u32m1(0x00040008, 8));
            vuint16m1_t ಸೂಚ್ಯಂಕ = __riscv_vor_vv_u16m1(qsb, __riscv_vand_vx_u16m1(__riscv_vsll_vv_u16m1(qhb, ಸರಿಸು, 16), 0x700, 16), 16);
            ಸೂಚ್ಯಂಕ = __riscv_vsll_vx_u16m1(ಸೂಚ್ಯಂಕ, 3, 16);
            qs += 16;

            __asm__ __volatile__("" ::: "memory");

            // Load the grid.
            ಸ್ಥಿರಾಂಕ vint8m4_t iq1b = __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vreinterpret_v_u64m4_i64m4(
                __riscv_vluxei16_v_u64m4(iq1s_grid, ಸೂಚ್ಯಂಕ, 16)));

            // Prepare the deltas.
            ಸ್ಥಿರಾಂಕ vbool16_t ಮುಖವಾಡ = __riscv_vmsgtu_vx_u16m1_b16(
                __riscv_vand_vv_u16m1(qhb, __riscv_vreinterpret_v_u32m1_u16m1(__riscv_vmv_v_x_u32m1(0x00800008, 8)), 16), 0, 16);
            ಸ್ಥಿರಾಂಕ vint64m4_t delta_pos = __riscv_vmv_v_x_i64m4(0x0101010101010101, 16);
            ಸ್ಥಿರಾಂಕ vint8m4_t delta = __riscv_vreinterpret_v_i64m4_i8m4(
                __riscv_vmerge_vxm_i64m4(delta_pos, 0xffffffffffffffff, ಮುಖವಾಡ, 16));

            // Load q8 for sub-blocks.
            ಸ್ಥಿರಾಂಕ vint8m4_t q8b = __riscv_vle8_v_i8m4(q8, 128);
            q8 += 128;

            // Calculate the lsums.
            ಸ್ಥಿರಾಂಕ vint16m8_t lsum1 = __riscv_vwmul_vv_i16m8(iq1b, q8b, 128);
            ಸ್ಥಿರಾಂಕ vint16m8_t lsum2 = __riscv_vwmul_vv_i16m8(delta, q8b, 128);

            // Prepare the scales.
            ಸ್ಥಿರಾಂಕ int16_t ls_0_0 = 2*((sc[0] >> 0) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_0_1 = 2*((sc[0] >> 3) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_1_0 = 2*((sc[0] >> 6) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_1_1 = 2*((sc[0] >> 9) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_2_0 = 2*((sc[1] >> 0) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_2_1 = 2*((sc[1] >> 3) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_3_0 = 2*((sc[1] >> 6) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_3_1 = 2*((sc[1] >> 9) & 0x7) + 1;
            sc += 2;

            // Accumulate in acc0 and acc1 for each sub-block.
            acc1 = __riscv_vwmacc_vx_i32m2(acc1, ls_0_0, __riscv_vget_v_i16m8_i16m1(lsum1, 0), 16);
            acc1 = __riscv_vwmacc_vx_i32m2(acc1, ls_0_1, __riscv_vget_v_i16m8_i16m1(lsum1, 1), 16);
            acc2 = __riscv_vwmacc_vx_i32m2(acc2, ls_0_0, __riscv_vget_v_i16m8_i16m1(lsum2, 0), 16);
            acc2 = __riscv_vwmacc_vx_i32m2(acc2, ls_0_1, __riscv_vget_v_i16m8_i16m1(lsum2, 1), 16);
            //
            acc1 = __riscv_vwmacc_vx_i32m2(acc1, ls_1_0, __riscv_vget_v_i16m8_i16m1(lsum1, 2), 16);
            acc1 = __riscv_vwmacc_vx_i32m2(acc1, ls_1_1, __riscv_vget_v_i16m8_i16m1(lsum1, 3), 16);
            acc2 = __riscv_vwmacc_vx_i32m2(acc2, ls_1_0, __riscv_vget_v_i16m8_i16m1(lsum2, 2), 16);
            acc2 = __riscv_vwmacc_vx_i32m2(acc2, ls_1_1, __riscv_vget_v_i16m8_i16m1(lsum2, 3), 16);
            //
            acc1 = __riscv_vwmacc_vx_i32m2(acc1, ls_2_0, __riscv_vget_v_i16m8_i16m1(lsum1, 4), 16);
            acc1 = __riscv_vwmacc_vx_i32m2(acc1, ls_2_1, __riscv_vget_v_i16m8_i16m1(lsum1, 5), 16);
            acc2 = __riscv_vwmacc_vx_i32m2(acc2, ls_2_0, __riscv_vget_v_i16m8_i16m1(lsum2, 4), 16);
            acc2 = __riscv_vwmacc_vx_i32m2(acc2, ls_2_1, __riscv_vget_v_i16m8_i16m1(lsum2, 5), 16);
            //
            acc1 = __riscv_vwmacc_vx_i32m2(acc1, ls_3_0, __riscv_vget_v_i16m8_i16m1(lsum1, 6), 16);
            acc1 = __riscv_vwmacc_vx_i32m2(acc1, ls_3_1, __riscv_vget_v_i16m8_i16m1(lsum1, 7), 16);
            acc2 = __riscv_vwmacc_vx_i32m2(acc2, ls_3_0, __riscv_vget_v_i16m8_i16m1(lsum2, 6), 16);
            acc2 = __riscv_vwmacc_vx_i32m2(acc2, ls_3_1, __riscv_vget_v_i16m8_i16m1(lsum2, 7), 16);

            __asm__ __volatile__("" ::: "memory");
        }

        // Reduce and accumulate in `sumf`.
        vint32m1_t one = __riscv_vmv_v_x_i32m1(0, 1);
        ಪೂರ್ಣಾಂಕ sumi1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m2_i32m1(acc1, one, 16));
        ಪೂರ್ಣಾಂಕ sumi2 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m2_i32m1(acc2, one, 16));
        sumf += y[i].d * GGML_CPU_FP16_ಗಮ್ಯ_FP32(scale.f16) * (sumi1 + IQ1M_DELTA * sumi2);
    }

    *s = sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq1_m_q8_K_vl512(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq1_m * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K  * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    iq1m_scale_t scale;

    // Mask for processing 16 elements per lsum register.
    ಸ್ಥಿರಾಂಕ vuint16m1_t l_ಸೂಚ್ಯಂಕ = __riscv_vid_v_u16m1(32);
    ಸ್ಥಿರಾಂಕ vbool16_t l_ಮುಖವಾಡ = __riscv_vmsgtu_vx_u16m1_b16(l_ಸೂಚ್ಯಂಕ, 15, 32);

    ದಶಮಾಂಶ sumf = 0.0f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ int8_t   * q8 = y[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t  * qs = x[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t  * qh = x[i].qh;
        ಸ್ಥಿರಾಂಕ uint16_t * sc = (ಸ್ಥಿರಾಂಕ uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        // Accumulators.
        vint32m2_t acc1 = __riscv_vmv_v_x_i32m2(0, 32);
        vint32m2_t acc2 = __riscv_vmv_v_x_i32m2(0, 32);

        // We process all the sub-blocks together.
        #ಸೂಚನೆ GCC unroll 1
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ib = 0; ib < QK_K/256; ib++) {
            // Load qh for all 16 sub-blocks.
            ಸ್ಥಿರಾಂಕ vuint8mf4_t qh_8 = __riscv_vle8_v_u8mf4(qh, 16);
            ಸ್ಥಿರಾಂಕ vuint16mf2_t qh_16_lo = __riscv_vzext_vf2_u16mf2(qh_8, 16);
            ಸ್ಥಿರಾಂಕ vuint16mf2_t qh_16_hi = __riscv_vsll_vx_u16mf2(qh_16_lo, 8, 16);
            ಸ್ಥಿರಾಂಕ vuint16m1_t qhb = __riscv_vzext_vf2_u16m1(
                __riscv_vreinterpret_v_u16mf2_u8mf2(__riscv_vor_vv_u16mf2(qh_16_lo, qh_16_hi, 16)), 32);
            __asm__ __volatile__("" ::: "memory");

            // Prepare grid indices.
            ಸ್ಥಿರಾಂಕ vuint16m1_t qsb = __riscv_vzext_vf2_u16m1(__riscv_vle8_v_u8mf2(&qs[0], 32), 32);
            ಸ್ಥಿರಾಂಕ vuint16m1_t ಸರಿಸು = __riscv_vreinterpret_v_u32m1_u16m1(__riscv_vmv_v_x_u32m1(0x00040008, 16));
            vuint16m1_t ಸೂಚ್ಯಂಕ = __riscv_vor_vv_u16m1(qsb, __riscv_vand_vx_u16m1(__riscv_vsll_vv_u16m1(qhb, ಸರಿಸು, 32), 0x700, 32), 32);
            ಸೂಚ್ಯಂಕ = __riscv_vsll_vx_u16m1(ಸೂಚ್ಯಂಕ, 3, 32);
            __asm__ __volatile__("" ::: "memory");

            // Load the grid.
            ಸ್ಥಿರಾಂಕ vint8m4_t iq1b = __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vreinterpret_v_u64m4_i64m4(
                __riscv_vluxei16_v_u64m4(iq1s_grid, ಸೂಚ್ಯಂಕ, 32)));

            // Prepare the deltas.
            ಸ್ಥಿರಾಂಕ vbool16_t ಮುಖವಾಡ = __riscv_vmsgtu_vx_u16m1_b16(
                __riscv_vand_vv_u16m1(qhb, __riscv_vreinterpret_v_u32m1_u16m1(__riscv_vmv_v_x_u32m1(0x00800008, 16)), 32), 0, 32);
            ಸ್ಥಿರಾಂಕ vint64m4_t delta_pos = __riscv_vmv_v_x_i64m4(0x0101010101010101, 32);
            ಸ್ಥಿರಾಂಕ vint8m4_t delta = __riscv_vreinterpret_v_i64m4_i8m4(
                __riscv_vmerge_vxm_i64m4(delta_pos, 0xffffffffffffffff, ಮುಖವಾಡ, 32));

            // Load q8 for sub-blocks.
            ಸ್ಥಿರಾಂಕ vint8m4_t q8b = __riscv_vle8_v_i8m4(q8, 256);

            // Calculate the lsums.
            ಸ್ಥಿರಾಂಕ vint16m8_t lsum1 = __riscv_vwmul_vv_i16m8(iq1b, q8b, 256);
            ಸ್ಥಿರಾಂಕ vint16m8_t lsum2 = __riscv_vwmul_vv_i16m8(delta, q8b, 256);

            // Prepare the scales.
            ಸ್ಥಿರಾಂಕ int16_t ls_0 = 2*((sc[0] >> 0) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_1 = 2*((sc[0] >> 3) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_2 = 2*((sc[0] >> 6) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_3 = 2*((sc[0] >> 9) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_4 = 2*((sc[1] >> 0) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_5 = 2*((sc[1] >> 3) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_6 = 2*((sc[1] >> 6) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_7 = 2*((sc[1] >> 9) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_8 = 2*((sc[2] >> 0) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_9 = 2*((sc[2] >> 3) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_10 = 2*((sc[2] >> 6) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_11 = 2*((sc[2] >> 9) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_12 = 2*((sc[3] >> 0) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_13 = 2*((sc[3] >> 3) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_14 = 2*((sc[3] >> 6) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_15 = 2*((sc[3] >> 9) & 0x7) + 1;

            // Accumulate in acc0 and acc1 for each sub-block.
            acc1 = __riscv_vwmacc_vx_i32m2(          acc1, ls_0, __riscv_vget_v_i16m8_i16m1(lsum1, 0), 16);
            acc1 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ, acc1, ls_1, __riscv_vget_v_i16m8_i16m1(lsum1, 0), 32);
            acc2 = __riscv_vwmacc_vx_i32m2(          acc2, ls_0, __riscv_vget_v_i16m8_i16m1(lsum2, 0), 16);
            acc2 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ, acc2, ls_1, __riscv_vget_v_i16m8_i16m1(lsum2, 0), 32);
            //
            acc1 = __riscv_vwmacc_vx_i32m2(          acc1, ls_2, __riscv_vget_v_i16m8_i16m1(lsum1, 1), 16);
            acc1 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ, acc1, ls_3, __riscv_vget_v_i16m8_i16m1(lsum1, 1), 32);
            acc2 = __riscv_vwmacc_vx_i32m2(          acc2, ls_2, __riscv_vget_v_i16m8_i16m1(lsum2, 1), 16);
            acc2 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ, acc2, ls_3, __riscv_vget_v_i16m8_i16m1(lsum2, 1), 32);
            //
            acc1 = __riscv_vwmacc_vx_i32m2(          acc1, ls_4, __riscv_vget_v_i16m8_i16m1(lsum1, 2), 16);
            acc1 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ, acc1, ls_5, __riscv_vget_v_i16m8_i16m1(lsum1, 2), 32);
            acc2 = __riscv_vwmacc_vx_i32m2(          acc2, ls_4, __riscv_vget_v_i16m8_i16m1(lsum2, 2), 16);
            acc2 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ, acc2, ls_5, __riscv_vget_v_i16m8_i16m1(lsum2, 2), 32);
            //
            acc1 = __riscv_vwmacc_vx_i32m2(          acc1, ls_6, __riscv_vget_v_i16m8_i16m1(lsum1, 3), 16);
            acc1 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ, acc1, ls_7, __riscv_vget_v_i16m8_i16m1(lsum1, 3), 32);
            acc2 = __riscv_vwmacc_vx_i32m2(          acc2, ls_6, __riscv_vget_v_i16m8_i16m1(lsum2, 3), 16);
            acc2 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ, acc2, ls_7, __riscv_vget_v_i16m8_i16m1(lsum2, 3), 32);
            //
            acc1 = __riscv_vwmacc_vx_i32m2(          acc1, ls_8, __riscv_vget_v_i16m8_i16m1(lsum1, 4), 16);
            acc1 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ, acc1, ls_9, __riscv_vget_v_i16m8_i16m1(lsum1, 4), 32);
            acc2 = __riscv_vwmacc_vx_i32m2(          acc2, ls_8, __riscv_vget_v_i16m8_i16m1(lsum2, 4), 16);
            acc2 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ, acc2, ls_9, __riscv_vget_v_i16m8_i16m1(lsum2, 4), 32);
            //
            acc1 = __riscv_vwmacc_vx_i32m2(          acc1, ls_10, __riscv_vget_v_i16m8_i16m1(lsum1, 5), 16);
            acc1 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ, acc1, ls_11, __riscv_vget_v_i16m8_i16m1(lsum1, 5), 32);
            acc2 = __riscv_vwmacc_vx_i32m2(          acc2, ls_10, __riscv_vget_v_i16m8_i16m1(lsum2, 5), 16);
            acc2 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ, acc2, ls_11, __riscv_vget_v_i16m8_i16m1(lsum2, 5), 32);
            //
            acc1 = __riscv_vwmacc_vx_i32m2(          acc1, ls_12, __riscv_vget_v_i16m8_i16m1(lsum1, 6), 16);
            acc1 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ, acc1, ls_13, __riscv_vget_v_i16m8_i16m1(lsum1, 6), 32);
            acc2 = __riscv_vwmacc_vx_i32m2(          acc2, ls_12, __riscv_vget_v_i16m8_i16m1(lsum2, 6), 16);
            acc2 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ, acc2, ls_13, __riscv_vget_v_i16m8_i16m1(lsum2, 6), 32);
            //
            acc1 = __riscv_vwmacc_vx_i32m2(          acc1, ls_14, __riscv_vget_v_i16m8_i16m1(lsum1, 7), 16);
            acc1 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ, acc1, ls_15, __riscv_vget_v_i16m8_i16m1(lsum1, 7), 32);
            acc2 = __riscv_vwmacc_vx_i32m2(          acc2, ls_14, __riscv_vget_v_i16m8_i16m1(lsum2, 7), 16);
            acc2 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ, acc2, ls_15, __riscv_vget_v_i16m8_i16m1(lsum2, 7), 32);

            __asm__ __volatile__("" ::: "memory");
        }

        // Reduce and accumulate in `sumf`.
        vint32m1_t one = __riscv_vmv_v_x_i32m1(0, 1);
        ಪೂರ್ಣಾಂಕ sumi1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m2_i32m1(acc1, one, 32));
        ಪೂರ್ಣಾಂಕ sumi2 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m2_i32m1(acc2, one, 32));
        sumf += y[i].d * GGML_CPU_FP16_ಗಮ್ಯ_FP32(scale.f16) * (sumi1 + IQ1M_DELTA * sumi2);
    }

    *s = sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq1_m_q8_K_vl1024(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq1_m * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K  * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    iq1m_scale_t scale;
    ದಶಮಾಂಶ sumf = 0.0f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ int8_t   * q8 = y[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t  * qs = x[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t  * qh = x[i].qh;
        ಸ್ಥಿರಾಂಕ uint16_t * sc = (ಸ್ಥಿರಾಂಕ uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        // Accumulators.
        vint32m2_t acc1 = __riscv_vmv_v_x_i32m2(0, 64);
        vint32m2_t acc2 = __riscv_vmv_v_x_i32m2(0, 64);

        // We process all the sub-blocks together.
        #ಸೂಚನೆ GCC unroll 1
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ib = 0; ib < QK_K/256; ib++) {
            // Load qh for all 16 sub-blocks.
            ಸ್ಥಿರಾಂಕ vuint8mf8_t qh_8 = __riscv_vle8_v_u8mf8(qh, 16);
            ಸ್ಥಿರಾಂಕ vuint16mf4_t qh_16_lo = __riscv_vzext_vf2_u16mf4(qh_8, 16);
            ಸ್ಥಿರಾಂಕ vuint16mf4_t qh_16_hi = __riscv_vsll_vx_u16mf4(qh_16_lo, 8, 16);
            ಸ್ಥಿರಾಂಕ vuint16mf2_t qhb = __riscv_vzext_vf2_u16mf2(
                __riscv_vreinterpret_v_u16mf4_u8mf4(__riscv_vor_vv_u16mf4(qh_16_lo, qh_16_hi, 16)), 32);
            __asm__ __volatile__("" ::: "memory");

            // Prepare grid indices.
            ಸ್ಥಿರಾಂಕ vuint16mf2_t qsb = __riscv_vzext_vf2_u16mf2(__riscv_vle8_v_u8mf4(&qs[0], 32), 32);
            ಸ್ಥಿರಾಂಕ vuint16mf2_t ಸರಿಸು = __riscv_vreinterpret_v_u32mf2_u16mf2(__riscv_vmv_v_x_u32mf2(0x00040008, 16));
            vuint16mf2_t ಸೂಚ್ಯಂಕ = __riscv_vor_vv_u16mf2(qsb, __riscv_vand_vx_u16mf2(__riscv_vsll_vv_u16mf2(qhb, ಸರಿಸು, 32), 0x700, 32), 32);
            ಸೂಚ್ಯಂಕ = __riscv_vsll_vx_u16mf2(ಸೂಚ್ಯಂಕ, 3, 32);
            __asm__ __volatile__("" ::: "memory");

            // Load the grid.
            ಸ್ಥಿರಾಂಕ vint8m2_t iq1b = __riscv_vreinterpret_v_i64m2_i8m2(__riscv_vreinterpret_v_u64m2_i64m2(
                __riscv_vluxei16_v_u64m2(iq1s_grid, ಸೂಚ್ಯಂಕ, 32)));

            // Prepare the deltas.
            ಸ್ಥಿರಾಂಕ vbool32_t ಮುಖವಾಡ = __riscv_vmsgtu_vx_u16mf2_b32(
                __riscv_vand_vv_u16mf2(qhb, __riscv_vreinterpret_v_u32mf2_u16mf2(__riscv_vmv_v_x_u32mf2(0x00800008, 16)), 32), 0, 32);
            ಸ್ಥಿರಾಂಕ vint64m2_t delta_pos = __riscv_vmv_v_x_i64m2(0x0101010101010101, 32);
            ಸ್ಥಿರಾಂಕ vint8m2_t delta = __riscv_vreinterpret_v_i64m2_i8m2(
                __riscv_vmerge_vxm_i64m2(delta_pos, 0xffffffffffffffff, ಮುಖವಾಡ, 32));

            // Load q8 for sub-blocks.
            ಸ್ಥಿರಾಂಕ vint8m2_t q8b = __riscv_vle8_v_i8m2(q8, 256);

            // Calculate the lsums.
            ಸ್ಥಿರಾಂಕ vint16m4_t lsum1 = __riscv_vwmul_vv_i16m4(iq1b, q8b, 256);
            ಸ್ಥಿರಾಂಕ vint16m4_t lsum2 = __riscv_vwmul_vv_i16m4(delta, q8b, 256);

            // Prepare the scales.
            ಸ್ಥಿರಾಂಕ int16_t ls_0 = 2*((sc[0] >> 0) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_1 = 2*((sc[0] >> 3) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_2 = 2*((sc[0] >> 6) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_3 = 2*((sc[0] >> 9) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_4 = 2*((sc[1] >> 0) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_5 = 2*((sc[1] >> 3) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_6 = 2*((sc[1] >> 6) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_7 = 2*((sc[1] >> 9) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_8 = 2*((sc[2] >> 0) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_9 = 2*((sc[2] >> 3) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_10 = 2*((sc[2] >> 6) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_11 = 2*((sc[2] >> 9) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_12 = 2*((sc[3] >> 0) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_13 = 2*((sc[3] >> 3) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_14 = 2*((sc[3] >> 6) & 0x7) + 1;
            ಸ್ಥಿರಾಂಕ int16_t ls_15 = 2*((sc[3] >> 9) & 0x7) + 1;

            // Mask for processing 16 elements per lsum register.
            ಸ್ಥಿರಾಂಕ vuint16m1_t l_ಸೂಚ್ಯಂಕ = __riscv_vid_v_u16m1(64);

            // Accumulate in acc1 and acc2 for each sub-block.
            acc1 = __riscv_vwmacc_vx_i32m2(acc1, ls_0,  __riscv_vget_v_i16m4_i16m1(lsum1, 0), 16);
            acc2 = __riscv_vwmacc_vx_i32m2(acc2, ls_0,  __riscv_vget_v_i16m4_i16m1(lsum2, 0), 16);
            acc1 = __riscv_vwmacc_vx_i32m2(acc1, ls_4,  __riscv_vget_v_i16m4_i16m1(lsum1, 1), 16);
            acc2 = __riscv_vwmacc_vx_i32m2(acc2, ls_4,  __riscv_vget_v_i16m4_i16m1(lsum2, 1), 16);
            acc1 = __riscv_vwmacc_vx_i32m2(acc1, ls_8,  __riscv_vget_v_i16m4_i16m1(lsum1, 2), 16);
            acc2 = __riscv_vwmacc_vx_i32m2(acc2, ls_8,  __riscv_vget_v_i16m4_i16m1(lsum2, 2), 16);
            acc1 = __riscv_vwmacc_vx_i32m2(acc1, ls_12, __riscv_vget_v_i16m4_i16m1(lsum1, 3), 16);
            acc2 = __riscv_vwmacc_vx_i32m2(acc2, ls_12, __riscv_vget_v_i16m4_i16m1(lsum2, 3), 16);
            //
            ಸ್ಥಿರಾಂಕ vbool16_t l_ಮುಖವಾಡ_16_32 = __riscv_vmsgtu_vx_u16m1_b16(l_ಸೂಚ್ಯಂಕ, 15, 64);
            acc1 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ_16_32, acc1, ls_1, __riscv_vget_v_i16m4_i16m1(lsum1, 0), 32);
            acc2 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ_16_32, acc2, ls_1, __riscv_vget_v_i16m4_i16m1(lsum2, 0), 32);
            acc1 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ_16_32, acc1, ls_5, __riscv_vget_v_i16m4_i16m1(lsum1, 1), 32);
            acc2 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ_16_32, acc2, ls_5, __riscv_vget_v_i16m4_i16m1(lsum2, 1), 32);
            acc1 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ_16_32, acc1, ls_9, __riscv_vget_v_i16m4_i16m1(lsum1, 2), 32);
            acc2 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ_16_32, acc2, ls_9, __riscv_vget_v_i16m4_i16m1(lsum2, 2), 32);
            acc1 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ_16_32, acc1, ls_13, __riscv_vget_v_i16m4_i16m1(lsum1, 3), 32);
            acc2 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ_16_32, acc2, ls_13, __riscv_vget_v_i16m4_i16m1(lsum2, 3), 32);
            //
            ಸ್ಥಿರಾಂಕ vbool16_t l_ಮುಖವಾಡ_32_48 = __riscv_vmsgtu_vx_u16m1_b16(l_ಸೂಚ್ಯಂಕ, 31, 64);
            acc1 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ_32_48, acc1, ls_2,  __riscv_vget_v_i16m4_i16m1(lsum1, 0), 48);
            acc2 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ_32_48, acc2, ls_2,  __riscv_vget_v_i16m4_i16m1(lsum2, 0), 48);
            acc1 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ_32_48, acc1, ls_6,  __riscv_vget_v_i16m4_i16m1(lsum1, 1), 48);
            acc2 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ_32_48, acc2, ls_6,  __riscv_vget_v_i16m4_i16m1(lsum2, 1), 48);
            acc1 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ_32_48, acc1, ls_10, __riscv_vget_v_i16m4_i16m1(lsum1, 2), 48);
            acc2 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ_32_48, acc2, ls_10, __riscv_vget_v_i16m4_i16m1(lsum2, 2), 48);
            acc1 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ_32_48, acc1, ls_14, __riscv_vget_v_i16m4_i16m1(lsum1, 3), 48);
            acc2 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ_32_48, acc2, ls_14, __riscv_vget_v_i16m4_i16m1(lsum2, 3), 48);
            //
            ಸ್ಥಿರಾಂಕ vbool16_t l_ಮುಖವಾಡ_48_64 = __riscv_vmsgtu_vx_u16m1_b16(l_ಸೂಚ್ಯಂಕ, 47, 64);
            acc1 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ_48_64, acc1, ls_3,  __riscv_vget_v_i16m4_i16m1(lsum1, 0), 64);
            acc2 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ_48_64, acc2, ls_3,  __riscv_vget_v_i16m4_i16m1(lsum2, 0), 64);
            acc1 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ_48_64, acc1, ls_7,  __riscv_vget_v_i16m4_i16m1(lsum1, 1), 64);
            acc2 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ_48_64, acc2, ls_7,  __riscv_vget_v_i16m4_i16m1(lsum2, 1), 64);
            acc1 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ_48_64, acc1, ls_11, __riscv_vget_v_i16m4_i16m1(lsum1, 2), 64);
            acc2 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ_48_64, acc2, ls_11, __riscv_vget_v_i16m4_i16m1(lsum2, 2), 64);
            acc1 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ_48_64, acc1, ls_15, __riscv_vget_v_i16m4_i16m1(lsum1, 3), 64);
            acc2 = __riscv_vwmacc_vx_i32m2_m(l_ಮುಖವಾಡ_48_64, acc2, ls_15, __riscv_vget_v_i16m4_i16m1(lsum2, 3), 64);

            __asm__ __volatile__("" ::: "memory");
        }

        // Reduce and accumulate in `sumf`.
        vint32m1_t one = __riscv_vmv_v_x_i32m1(0, 1);
        ಪೂರ್ಣಾಂಕ sumi1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m2_i32m1(acc1, one, 64));
        ಪೂರ್ಣಾಂಕ sumi2 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m2_i32m1(acc2, one, 64));
        sumf += y[i].d * GGML_CPU_FP16_ಗಮ್ಯ_FP32(scale.f16) * (sumi1 + IQ1M_DELTA * sumi2);
    }

    *s = sumf;
}
#ಆದರೆ_ಅಂತ್ಯ

ಶೂನ್ಯ ggml_vec_dot_iq1_m_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
#ಆದರೆ defined __riscv_v
    ಆಯ್ಕೆ (__riscv_vlenb() * 8) {
        ಸ್ಥಿತಿ 128:
            ggml_vec_dot_iq1_m_q8_K_vl128(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 256:
            ggml_vec_dot_iq1_m_q8_K_vl256(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 512:
            ggml_vec_dot_iq1_m_q8_K_vl512(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 1024:
            ggml_vec_dot_iq1_m_q8_K_vl1024(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ggml_vec_dot_iq1_m_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ggml_vec_dot_iq1_m_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

#ಆದರೆ defined __riscv_v
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint8_t sign_gather_indices_arr[64] = {
    0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3,
    4,4,4,4,4,4,4,4, 5,5,5,5,5,5,5,5, 6,6,6,6,6,6,6,6, 7,7,7,7,7,7,7,7
};

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint8_t sign_ಬಿಟ್_masks_arr[64] = {
    1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128,
    1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128
};

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq2_s_q8_K_vl128(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಬಳಸದ(nrc); ಬಳಸದ(bx); ಬಳಸದ(by); ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_s * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K  * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;
    ಸ್ಥಿರಾಂಕ uint64_t * grid64 = (ಸ್ಥಿರಾಂಕ uint64_t *)iq2s_grid;

    // Pre-load Constants
    vuint8m2_t v_ids = __riscv_vid_v_u8m2(32);
    vuint8m2_t v_sign_gather_indices = __riscv_vsrl_vx_u8m2(v_ids, 3, 32);
    vuint8m2_t v_ones = __riscv_vmv_v_x_u8m2(1, 32);
    vuint8m2_t v_ಸರಿಸು_amts = __riscv_vand_vx_u8m2(v_ids, 7, 32);
    vuint8m2_t v_sign_masks = __riscv_vsll_vv_u8m2(v_ones, v_ಸರಿಸು_amts, 32);
    uint16_t ಸರಿಸು_qh_arr[4] = {11, 9, 7, 5};
    vuint16mf2_t v_ಸರಿಸು_qh = __riscv_vle16_v_u16mf2(ಸರಿಸು_qh_arr, 4);

    ದಶಮಾಂಶ sumf = 0.0f;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ combined_scale = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d;

        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT qs = x[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT qh = x[i].qh;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT scales = x[i].scales;
        ಸ್ಥಿರಾಂಕ int8_t  * GGML_RESTRICT q8 = y[i].qs;

        ಸ್ಥಿರಾಂಕ uint8_t * signs_ptr = qs + 32;
        ದಶಮಾಂಶ ಮೊತ್ತ_ಖಂಡ = 0.0f;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ib = 0; ib < 8; ++ib) {

            // Load Low Bits [4 bytes]
            vuint8mf4_t v_qs_u8 = __riscv_vle8_v_u8mf4(qs, 4);
            qs += 4;

            // Load 1 byte. It contains bits for 4 mini-blocks.
            uint8_t qh_val = *qh++;

            // Combine Low + High bits of 10bit indices
            vuint8mf4_t v_qh_raw = __riscv_vmv_v_x_u8mf4(qh_val, 4);
            vuint16mf2_t v_qh_u16 = __riscv_vwcvtu_x_x_v_u16mf2(v_qh_raw, 4);
            vuint16mf2_t v_qh_mf2 = __riscv_vsll_vv_u16mf2(v_qh_u16, v_ಸರಿಸು_qh, 4);
            v_qh_mf2 = __riscv_vand_vx_u16mf2(v_qh_mf2, 0x1800, 4);
            vuint16mf2_t v_qs_u16_mf2 = __riscv_vwcvtu_x_x_v_u16mf2(v_qs_u8, 4);
            vuint16mf2_t v_qs_u16 = __riscv_vsll_vx_u16mf2(v_qs_u16_mf2, 3, 4);
            vuint16mf2_t v_grid_offsets = __riscv_vor_vv_u16mf2(v_qs_u16, v_qh_mf2, 4);

            // Lookup Grid
            vint8m2_t v_grid_i8 = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vreinterpret_v_u64m2_u8m2(__riscv_vluxei16_v_u64m2(grid64, v_grid_offsets, 4)));

            vuint8mf4_t v_signs_raw = __riscv_vle8_v_u8mf4(signs_ptr, 4);
            signs_ptr += 4;
            vuint8m2_t v_signs_ಮೂಲ = __riscv_vlmul_ext_v_u8mf4_u8m2(v_signs_raw);
            vuint8m2_t v_signs_bcast = __riscv_vrgather_vv_u8m2(v_signs_ಮೂಲ, v_sign_gather_indices, 32);

            // generating sign mask
            vuint8m2_t v_sign_ಬಿಟ್ಟುಗಳು = __riscv_vand_vv_u8m2(v_signs_bcast, v_sign_masks, 32);
            vbool4_t m_negative = __riscv_vmsne_vx_u8m2_b4(v_sign_ಬಿಟ್ಟುಗಳು, 0, 32);

            vint8m2_t v_q8 = __riscv_vle8_v_i8m2(q8, 32);
            q8 += 32;

            // apply signs
            vint8m2_t v_q8_signed = __riscv_vrsub_vx_i8m2_mu(m_negative,v_q8, v_q8, 0, 32);
            vint16m4_t v_dot = __riscv_vwmul_vv_i16m4(v_grid_i8, v_q8_signed, 32);

            // Reduction
            vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, 1);

            // Reduce 0-15 (First Half)
            int32_t s0 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(
                __riscv_vget_v_i16m4_i16m2(v_dot, 0), v_zero, 16));

            // Reduce 16-31 (Second Half)
            int32_t s1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(
                __riscv_vget_v_i16m4_i16m2(v_dot, 1), v_zero, 16));

            // Apply sub Scales
            uint8_t sc = *scales++;

            ಮೊತ್ತ_ಖಂಡ += s0 * (2 * (sc & 0xF) + 1);
            ಮೊತ್ತ_ಖಂಡ += s1 * (2 * (sc >> 4)  + 1);
        }
        sumf += ಮೊತ್ತ_ಖಂಡ * combined_scale;
    }
    *s = 0.125f * sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq2_s_q8_K_vl256(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಬಳಸದ(nrc); ಬಳಸದ(bx); ಬಳಸದ(by); ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_s * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K  * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;
    ಸ್ಥಿರಾಂಕ uint64_t * grid64 = (ಸ್ಥಿರಾಂಕ uint64_t *)iq2s_grid;

    // --- Pre-load Constants ---
    uint16_t gather_qh_arr[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    vuint16mf2_t v_gather_qh = __riscv_vle16_v_u16mf2(gather_qh_arr, 8);
    uint16_t ಸರಿಸು_qh_arr[8] = {11, 9, 7, 5, 11, 9, 7, 5};
    vuint16mf2_t v_ಸರಿಸು_qh = __riscv_vle16_v_u16mf2(ಸರಿಸು_qh_arr, 8);

    // Constants for sign extraction
    vuint8m2_t v_sign_gather_indices = __riscv_vle8_v_u8m2(sign_gather_indices_arr, 64);
    vuint8m2_t v_sign_masks = __riscv_vle8_v_u8m2(sign_ಬಿಟ್_masks_arr, 64);

    ದಶಮಾಂಶ sumf = 0.0f;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ combined_scale = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d;

        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT qs = x[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT qh = x[i].qh;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT scales = x[i].scales;
        ಸ್ಥಿರಾಂಕ int8_t  * GGML_RESTRICT q8 = y[i].qs;

        ಸ್ಥಿರಾಂಕ uint8_t * signs_ptr = qs + 32;

        ದಶಮಾಂಶ ಮೊತ್ತ_ಖಂಡ = 0.0f;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ib = 0; ib < 4; ++ib) {
            // Combine low + high bits
            vuint8mf4_t v_qs_u8 = __riscv_vle8_v_u8mf4(qs, 8);
            qs += 8;
            uint16_t qh_val;
            ಸ್ಮರಣೆ_ನಕಲು(&qh_val, qh, 2);
            qh += 2;
            vuint8mf8_t v_qh_raw = __riscv_vle8_v_u8mf8((ಸ್ಥಿರಾಂಕ uint8_t*)&qh_val, 2);
            vuint16mf4_t v_qh_u16 = __riscv_vwcvtu_x_x_v_u16mf4(v_qh_raw, 2);
            vuint16mf2_t v_qh_u16_ext = __riscv_vlmul_ext_v_u16mf4_u16mf2(v_qh_u16);
            vuint16mf2_t v_qh_expanded = __riscv_vrgather_vv_u16mf2(v_qh_u16_ext, v_gather_qh, 8);
            v_qh_expanded = __riscv_vsll_vv_u16mf2(v_qh_expanded, v_ಸರಿಸು_qh, 8);

            // Mask: We want bits 11-12. 0x1800 = 0001 1000 0000 0000
            v_qh_expanded = __riscv_vand_vx_u16mf2(v_qh_expanded, 0x1800, 8);
            vuint16mf2_t v_qs_u16 = __riscv_vwcvtu_x_x_v_u16mf2(v_qs_u8, 8);

            // Multiply by 8 to get byte offset, instead of element offset
            v_qs_u16 = __riscv_vsll_vx_u16mf2(v_qs_u16, 3, 8);
            vuint16mf2_t v_grid_offsets = __riscv_vor_vv_u16mf2(v_qs_u16, v_qh_expanded, 8);

            // Lookup Grid using Byte Offsets
            vuint64m2_t v_grid_vals = __riscv_vluxei16_v_u64m2(grid64, v_grid_offsets, 8);

            vuint8m2_t v_grid_u8 = __riscv_vreinterpret_v_u64m2_u8m2(v_grid_vals);
            vint8m2_t v_grid_i8 = __riscv_vreinterpret_v_u8m2_i8m2(v_grid_u8);

            // Load signs and generate sign mask
            vuint8mf4_t v_signs_raw = __riscv_vle8_v_u8mf4(signs_ptr, 8);
            signs_ptr += 8;

            vuint8m2_t v_signs_ಮೂಲ = __riscv_vlmul_ext_v_u8mf4_u8m2(v_signs_raw);
            vuint8m2_t v_signs_bcast = __riscv_vrgather_vv_u8m2(v_signs_ಮೂಲ, v_sign_gather_indices, 64);

            vuint8m2_t v_sign_ಬಿಟ್ಟುಗಳು = __riscv_vand_vv_u8m2(v_signs_bcast, v_sign_masks, 64);
            vbool4_t m_negative = __riscv_vmsne_vx_u8m2_b4(v_sign_ಬಿಟ್ಟುಗಳು, 0, 64);

            vint8m2_t v_q8 = __riscv_vle8_v_i8m2(q8, 64);
            q8 += 64;

            vint8m2_t v_q8_signed = __riscv_vrsub_vx_i8m2_mu(m_negative, v_q8, v_q8, 0, 64);
            vint16m4_t v_dot = __riscv_vwmul_vv_i16m4(v_grid_i8, v_q8_signed, 64);

            vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, 1);

            int32_t s0 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(
                __riscv_vget_v_i16m4_i16m1(v_dot, 0), v_zero, 16));
            int32_t s1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(
                __riscv_vget_v_i16m4_i16m1(v_dot, 1), v_zero, 16));
            int32_t s2 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(
                __riscv_vget_v_i16m4_i16m1(v_dot, 2), v_zero, 16));
            int32_t s3 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(
                __riscv_vget_v_i16m4_i16m1(v_dot, 3), v_zero, 16));

            uint8_t sc0 = scales[0];
            uint8_t sc1 = scales[1];
            scales += 2;

            ಮೊತ್ತ_ಖಂಡ += s0 * (2 * (sc0 & 0xF) + 1);
            ಮೊತ್ತ_ಖಂಡ += s1 * (2 * (sc0 >> 4)  + 1);
            ಮೊತ್ತ_ಖಂಡ += s2 * (2 * (sc1 & 0xF) + 1);
            ಮೊತ್ತ_ಖಂಡ += s3 * (2 * (sc1 >> 4)  + 1);
        }
        sumf += ಮೊತ್ತ_ಖಂಡ * combined_scale;
    }
    *s = 0.125f * sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq2_s_q8_K_vl512(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಬಳಸದ(nrc); ಬಳಸದ(bx); ಬಳಸದ(by); ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_s * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K  * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;
    ಸ್ಥಿರಾಂಕ uint64_t * grid64 = (ಸ್ಥಿರಾಂಕ uint64_t *)iq2s_grid;

    vuint8m2_t v_ids = __riscv_vid_v_u8m2(128);
    vuint8m2_t v_sign_gather_indices = __riscv_vsrl_vx_u8m2(v_ids, 3, 128);

    vuint8m2_t v_ones = __riscv_vmv_v_x_u8m2(1, 128);
    vuint8m2_t v_ಸರಿಸು_amts = __riscv_vand_vx_u8m2(v_ids, 7, 128);
    vuint8m2_t v_sign_masks = __riscv_vsll_vv_u8m2(v_ones, v_ಸರಿಸು_amts, 128);

    uint16_t gather_qh_arr[16] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    vuint16mf2_t v_gather_qh = __riscv_vle16_v_u16mf2(gather_qh_arr, 16);

    uint16_t ಸರಿಸು_qh_arr[16] = {11, 9, 7, 5, 11, 9, 7, 5, 11, 9, 7, 5, 11, 9, 7, 5};
    vuint16mf2_t v_ಸರಿಸು_qh = __riscv_vle16_v_u16mf2(ಸರಿಸು_qh_arr, 16);

    // Masks for selecting lower/upper 16 lanes within a 32-lane i16m1 register
    vuint16m1_t v_ids16 = __riscv_vid_v_u16m1(32);
    vbool16_t m_hi16 = __riscv_vmsgeu_vx_u16m1_b16(v_ids16, 16, 32);
    ದಶಮಾಂಶ sumf = 0.0f;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ combined_scale = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d;

        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT qs = x[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT qh = x[i].qh;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT scales = x[i].scales;
        ಸ್ಥಿರಾಂಕ int8_t  * GGML_RESTRICT q8 = y[i].qs;

        ಸ್ಥಿರಾಂಕ uint8_t * signs_ptr = qs + 32;

        ದಶಮಾಂಶ ಮೊತ್ತ_ಖಂಡ = 0.0f;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ib = 0; ib < 2; ++ib) {
            vuint8mf4_t v_qs_u8 = __riscv_vle8_v_u8mf4(qs, 16);
            qs += 16;

            vuint8mf8_t v_qh_raw = __riscv_vle8_v_u8mf8(qh, 4);
            qh += 4;

            vuint16mf4_t v_qh_u16 = __riscv_vwcvtu_x_x_v_u16mf4(v_qh_raw, 4);
            vuint16mf2_t v_qh_u16_ext = __riscv_vlmul_ext_v_u16mf4_u16mf2(v_qh_u16);
            vuint16mf2_t v_qh_expanded = __riscv_vrgather_vv_u16mf2(v_qh_u16_ext, v_gather_qh, 16);
            v_qh_expanded = __riscv_vsll_vv_u16mf2(v_qh_expanded, v_ಸರಿಸು_qh, 16);
            v_qh_expanded = __riscv_vand_vx_u16mf2(v_qh_expanded, 0x1800, 16);

            vuint16mf2_t v_qs_u16 = __riscv_vwcvtu_x_x_v_u16mf2(v_qs_u8, 16);
            v_qs_u16 = __riscv_vsll_vx_u16mf2(v_qs_u16, 3, 16);

            vuint16mf2_t v_grid_offsets = __riscv_vor_vv_u16mf2(v_qs_u16, v_qh_expanded, 16);
            vuint64m2_t v_grid_vals = __riscv_vluxei16_v_u64m2(grid64, v_grid_offsets, 16);
            vuint8m2_t v_grid_u8 = __riscv_vreinterpret_v_u64m2_u8m2(v_grid_vals);
            vint8m2_t v_grid_i8 = __riscv_vreinterpret_v_u8m2_i8m2(v_grid_u8);

            vuint8mf4_t v_signs_raw = __riscv_vle8_v_u8mf4(signs_ptr, 16);
            signs_ptr += 16;

            vuint8m2_t v_signs_ಮೂಲ = __riscv_vlmul_ext_v_u8mf4_u8m2(v_signs_raw);
            vuint8m2_t v_signs_bcast = __riscv_vrgather_vv_u8m2(v_signs_ಮೂಲ, v_sign_gather_indices, 128);
            vuint8m2_t v_sign_ಬಿಟ್ಟುಗಳು = __riscv_vand_vv_u8m2(v_signs_bcast, v_sign_masks, 128);
            vbool4_t m_negative = __riscv_vmsne_vx_u8m2_b4(v_sign_ಬಿಟ್ಟುಗಳು, 0, 128);
            vint8m2_t v_q8 = __riscv_vle8_v_i8m2(q8, 128);
            q8 += 128;

            vint8m2_t v_q8_signed = __riscv_vrsub_vx_i8m2_mu(m_negative, v_q8, v_q8, 0, 128);
            vint16m4_t v_dot = __riscv_vwmul_vv_i16m4(v_grid_i8, v_q8_signed, 128);

            vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, 1);
            vint16m1_t v0 = __riscv_vget_v_i16m4_i16m1(v_dot, 0);
            vint16m1_t v1 = __riscv_vget_v_i16m4_i16m1(v_dot, 1);
            vint16m1_t v2 = __riscv_vget_v_i16m4_i16m1(v_dot, 2);
            vint16m1_t v3 = __riscv_vget_v_i16m4_i16m1(v_dot, 3);

            int32_t s0 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(v0, v_zero, 16));
            int32_t s1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1_m(m_hi16, v0, v_zero, 32));
            int32_t s2 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(v1, v_zero, 16));
            int32_t s3 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1_m(m_hi16, v1, v_zero, 32));
            int32_t s4 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(v2, v_zero, 16));
            int32_t s5 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1_m(m_hi16, v2, v_zero, 32));
            int32_t s6 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1( v3, v_zero, 16));
            int32_t s7 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1_m(m_hi16, v3, v_zero, 32));

            uint8_t sc0 = scales[0];
            uint8_t sc1 = scales[1];
            uint8_t sc2 = scales[2];
            uint8_t sc3 = scales[3];
            scales += 4;

            ಮೊತ್ತ_ಖಂಡ += s0 * (2 * (sc0 & 0xF) + 1);
            ಮೊತ್ತ_ಖಂಡ += s1 * (2 * (sc0 >> 4)  + 1);
            ಮೊತ್ತ_ಖಂಡ += s2 * (2 * (sc1 & 0xF) + 1);
            ಮೊತ್ತ_ಖಂಡ += s3 * (2 * (sc1 >> 4)  + 1);
            ಮೊತ್ತ_ಖಂಡ += s4 * (2 * (sc2 & 0xF) + 1);
            ಮೊತ್ತ_ಖಂಡ += s5 * (2 * (sc2 >> 4)  + 1);
            ಮೊತ್ತ_ಖಂಡ += s6 * (2 * (sc3 & 0xF) + 1);
            ಮೊತ್ತ_ಖಂಡ += s7 * (2 * (sc3 >> 4)  + 1);
        }

        sumf += ಮೊತ್ತ_ಖಂಡ * combined_scale;
    }
    *s = 0.125f * sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq2_s_q8_K_vl1024(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಬಳಸದ(nrc); ಬಳಸದ(bx); ಬಳಸದ(by); ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_s * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K  * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;
    ಸ್ಥಿರಾಂಕ uint64_t * grid64 = (ಸ್ಥಿರಾಂಕ uint64_t *)iq2s_grid;
    vuint8m2_t v_ids = __riscv_vid_v_u8m2(256);
    vuint8m2_t v_sign_gather_indices = __riscv_vsrl_vx_u8m2(v_ids, 3, 256);

    vuint8m2_t v_ones = __riscv_vmv_v_x_u8m2(1, 256);
    vuint8m2_t v_ಸರಿಸು_amts = __riscv_vand_vx_u8m2(v_ids, 7, 256);
    vuint8m2_t v_sign_masks = __riscv_vsll_vv_u8m2(v_ones, v_ಸರಿಸು_amts, 256);

    uint16_t gather_qh_arr[32] = {
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
        4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7
    };
    vuint16mf2_t v_gather_qh = __riscv_vle16_v_u16mf2(gather_qh_arr, 32);

    uint16_t ಸರಿಸು_qh_arr[32] = {
        11, 9, 7, 5, 11, 9, 7, 5, 11, 9, 7, 5, 11, 9, 7, 5,
        11, 9, 7, 5, 11, 9, 7, 5, 11, 9, 7, 5, 11, 9, 7, 5
    };
    vuint16mf2_t v_ಸರಿಸು_qh = __riscv_vle16_v_u16mf2(ಸರಿಸು_qh_arr, 32);

    // Masks for 4 groups of 16 lanes within a 64-lane i16m4 chunk
    vuint16m4_t v_ids64 = __riscv_vid_v_u16m4(64);
    vbool4_t m_g0 = __riscv_vmsltu_vx_u16m4_b4(v_ids64, 16, 64);
    vbool4_t m_g1 = __riscv_vmand_mm_b4(
        __riscv_vmsgeu_vx_u16m4_b4(v_ids64, 16, 64),
        __riscv_vmsltu_vx_u16m4_b4(v_ids64, 32, 64), 64);
    vbool4_t m_g2 = __riscv_vmand_mm_b4(
        __riscv_vmsgeu_vx_u16m4_b4(v_ids64, 32, 64),
        __riscv_vmsltu_vx_u16m4_b4(v_ids64, 48, 64), 64);
    vbool4_t m_g3 = __riscv_vmsgeu_vx_u16m4_b4(v_ids64, 48, 64);

    ದಶಮಾಂಶ sumf = 0.0f;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ combined_scale = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d;

        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT qs = x[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT qh = x[i].qh;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT scales = x[i].scales;
        ಸ್ಥಿರಾಂಕ int8_t  * GGML_RESTRICT q8 = y[i].qs;

        ಸ್ಥಿರಾಂಕ uint8_t * signs_ptr = qs + 32;

        ದಶಮಾಂಶ ಮೊತ್ತ_ಖಂಡ = 0.0f;

        vuint8mf4_t v_qs_u8 = __riscv_vle8_v_u8mf4(qs, 32);
        qs += 32;

        vuint8mf8_t v_qh_raw = __riscv_vle8_v_u8mf8(qh, 8);
        qh += 8;

        vuint16mf4_t v_qh_u16 = __riscv_vwcvtu_x_x_v_u16mf4(v_qh_raw, 8);
        vuint16mf2_t v_qh_u16_ext = __riscv_vlmul_ext_v_u16mf4_u16mf2(v_qh_u16);
        vuint16mf2_t v_qh_expanded = __riscv_vrgather_vv_u16mf2(v_qh_u16_ext, v_gather_qh, 32);
        v_qh_expanded = __riscv_vsll_vv_u16mf2(v_qh_expanded, v_ಸರಿಸು_qh, 32);
        v_qh_expanded = __riscv_vand_vx_u16mf2(v_qh_expanded, 0x1800, 32);

        vuint16mf2_t v_qs_u16 = __riscv_vwcvtu_x_x_v_u16mf2(v_qs_u8, 32);
        v_qs_u16 = __riscv_vsll_vx_u16mf2(v_qs_u16, 3, 32);

        vuint16mf2_t v_grid_offsets = __riscv_vor_vv_u16mf2(v_qs_u16, v_qh_expanded, 32);
        vuint64m2_t v_grid_vals = __riscv_vluxei16_v_u64m2(grid64, v_grid_offsets, 32);
        vuint8m2_t v_grid_u8 = __riscv_vreinterpret_v_u64m2_u8m2(v_grid_vals);
        vint8m2_t v_grid_i8 = __riscv_vreinterpret_v_u8m2_i8m2(v_grid_u8);

        //loading signs
        vuint8mf2_t v_signs_raw = __riscv_vle8_v_u8mf2(signs_ptr, 32);
        signs_ptr += 32;

        vuint8m2_t v_signs_ಮೂಲ = __riscv_vlmul_ext_v_u8mf2_u8m2(v_signs_raw);
        vuint8m2_t v_signs_bcast = __riscv_vrgather_vv_u8m2(v_signs_ಮೂಲ, v_sign_gather_indices, 256);
        vuint8m2_t v_sign_ಬಿಟ್ಟುಗಳು = __riscv_vand_vv_u8m2(v_signs_bcast, v_sign_masks, 256);
        vbool4_t m_negative = __riscv_vmsne_vx_u8m2_b4(v_sign_ಬಿಟ್ಟುಗಳು, 0, 256);

        vint8m2_t v_q8 = __riscv_vle8_v_i8m2(q8, 256);
        q8 += 256;

        vint8m2_t v_q8_signed = __riscv_vrsub_vx_i8m2_mu(m_negative, v_q8, v_q8, 0, 256);
        vint16m4_t v_dot = __riscv_vwmul_vv_i16m4(v_grid_i8, v_q8_signed, 256);

        vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, 1);

        vint16m4_t c = v_dot;

        int32_t s0  = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(m_g0, c, v_zero, 64));
        int32_t s1  = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(m_g1, c, v_zero, 64));
        int32_t s2  = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(m_g2, c, v_zero, 64));
        int32_t s3  = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(m_g3, c, v_zero, 64));

        c = __riscv_vslidedown_vx_i16m4(c, 64, 256);
        int32_t s4  = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(m_g0, c, v_zero, 64));
        int32_t s5  = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(m_g1, c, v_zero, 64));
        int32_t s6  = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(m_g2, c, v_zero, 64));
        int32_t s7  = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(m_g3, c, v_zero, 64));

        c = __riscv_vslidedown_vx_i16m4(c, 64, 256);
        int32_t s8  = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(m_g0, c, v_zero, 64));
        int32_t s9  = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(m_g1, c, v_zero, 64));
        int32_t s10 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(m_g2, c, v_zero, 64));
        int32_t s11 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(m_g3, c, v_zero, 64));

        c = __riscv_vslidedown_vx_i16m4(c, 64, 256);
        int32_t s12 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(m_g0, c, v_zero, 64));
        int32_t s13 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(m_g1, c, v_zero, 64));
        int32_t s14 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(m_g2, c, v_zero, 64));
        int32_t s15 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(m_g3, c, v_zero, 64));

        int32_t sums_arr[16] = { s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15 };

        // Load 8 scale bytes and split into 16 nibbles
        vuint8mf2_t v_sc8 = __riscv_vle8_v_u8mf2(scales, 8);
        scales += 8;

        vuint8mf2_t v_lo8 = __riscv_vand_vx_u8mf2(v_sc8, 0x0F, 8);
        vuint8mf2_t v_hi8 = __riscv_vsrl_vx_u8mf2(v_sc8, 4, 8);

        vuint8m1_t v_idx16 = __riscv_vid_v_u8m1(16);
        vuint8m1_t v_half = __riscv_vsrl_vx_u8m1(v_idx16, 1, 16);
        vbool8_t m_even = __riscv_vmseq_vx_u8m1_b8(__riscv_vand_vx_u8m1(v_idx16, 1, 16), 0, 16);

        vuint8m1_t v_lo_ext = __riscv_vlmul_ext_v_u8mf2_u8m1(v_lo8);
        vuint8m1_t v_hi_ext = __riscv_vlmul_ext_v_u8mf2_u8m1(v_hi8);
        vuint8m1_t v_lo_g = __riscv_vrgather_vv_u8m1(v_lo_ext, v_half, 16);
        vuint8m1_t v_hi_g = __riscv_vrgather_vv_u8m1(v_hi_ext, v_half, 16);
        vuint8m1_t v_nib = __riscv_vmerge_vvm_u8m1(v_lo_g, v_hi_g, m_even, 16);

        ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint8_t iq2s_scale_lut_16_local[16] = {
            1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
        };
        vuint8m1_t v_lut = __riscv_vle8_v_u8m1(iq2s_scale_lut_16_local, 16);
        vuint8m1_t v_sc8v = __riscv_vrgather_vv_u8m1(v_lut, v_nib, 16);

        vint32m4_t v_sums = __riscv_vle32_v_i32m4(sums_arr, 16);
        vuint16m2_t v_sc16 = __riscv_vwcvtu_x_x_v_u16m2(v_sc8v, 16);
        vuint32m4_t v_sc32u = __riscv_vwcvtu_x_x_v_u32m4(v_sc16, 16);
        vint32m4_t v_sc32 = __riscv_vreinterpret_v_u32m4_i32m4(v_sc32u);
        vint32m4_t v_prod = __riscv_vmul_vv_i32m4(v_sums, v_sc32, 16);

        vint32m1_t v_zero32 = __riscv_vmv_v_x_i32m1(0, 1);
        int32_t ಮೊತ್ತ_part = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(v_prod, v_zero32, 16));
        ಮೊತ್ತ_ಖಂಡ += ಮೊತ್ತ_part;

        sumf += ಮೊತ್ತ_ಖಂಡ * combined_scale;
    }
    *s = 0.125f * sumf;
}
#ಆದರೆ_ಅಂತ್ಯ

ಶೂನ್ಯ ggml_vec_dot_iq2_s_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
#ಆದರೆ defined __riscv_v
    ಆಯ್ಕೆ (__riscv_vlenb() * 8) {
        ಸ್ಥಿತಿ 128:
            ggml_vec_dot_iq2_s_q8_K_vl128(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 256:
            ggml_vec_dot_iq2_s_q8_K_vl256(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 512:
            ggml_vec_dot_iq2_s_q8_K_vl512(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ggml_vec_dot_iq2_s_q8_K_vl1024(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ggml_vec_dot_iq2_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

#ಆದರೆ defined __riscv_v
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ int8_t keven_signs_q2xs[1024] = {
     1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1,
     1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1, -1,
     1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1, -1,
     1,  1, -1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
     1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1, -1,
     1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1,  1,
     1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1,  1,
     1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1, -1,
     1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1, -1,
     1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1,  1,
     1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1,  1,
     1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1, -1,
     1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1,  1,
     1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1, -1,
     1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1, -1,
     1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,
     1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1, -1,
     1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1,  1,
     1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1,  1,
     1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1, -1,
     1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1,  1,
     1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1, -1,
     1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1, -1,
     1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1,  1,
     1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1,  1,
     1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1, -1,
     1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1, -1,
     1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1,  1,
     1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1, -1,
     1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1,  1,
     1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1,  1,
     1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1,
};

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq2_xs_q8_K_vl128(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_xs * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K   * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;
    ಸ್ಥಿರಾಂಕ uint64_t * signs64 = (ಸ್ಥಿರಾಂಕ uint64_t *)keven_signs_q2xs;
    ಸ್ಥಿರಾಂಕ uint64_t * grid64  = (ಸ್ಥಿರಾಂಕ uint64_t *)iq2xs_grid;

    ದಶಮಾಂಶ sumf = 0.0f;
#ಸೂಚನೆ GCC unroll 1
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d;
        ಸ್ಥಿರಾಂಕ uint16_t * GGML_RESTRICT qs = x[i].qs;
        ಸ್ಥಿರಾಂಕ int8_t   * GGML_RESTRICT q8 = y[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t  * GGML_RESTRICT scales = x[i].scales;

        int32_t ಮೊತ್ತ_int = 0;

        // Loop over 4 subblocks of 64 elements
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ib64 = 0; ib64 < QK_K / 64; ++ib64) {

            // Load indices.
            vuint16m1_t v_qs = __riscv_vle16_v_u16m1(qs, 8);
            qs += 8;

            // Prepare offsets
            vuint16m1_t vidx_grid = __riscv_vsll_vx_u16m1(__riscv_vand_vx_u16m1(v_qs, 511, 8), 3, 8);
            vuint16m1_t vidx_sign = __riscv_vsll_vx_u16m1(__riscv_vsrl_vx_u16m1(v_qs, 9, 8), 3, 8);

            // load values and signs from the lookup tables
            vuint64m4_t vq2_64 = __riscv_vluxei16_v_u64m4(grid64, vidx_grid, 8);
            vuint64m4_t vs2_64 = __riscv_vluxei16_v_u64m4(signs64, vidx_sign, 8);
            vint8m4_t q2u = __riscv_vreinterpret_v_u8m4_i8m4(__riscv_vreinterpret_v_u64m4_u8m4(vq2_64));
            vint8m4_t q2s = __riscv_vreinterpret_v_u8m4_i8m4(__riscv_vreinterpret_v_u64m4_u8m4(vs2_64));
            vint8m4_t q2_final = __riscv_vmul_vv_i8m4(q2u, q2s, 64);
            asm ಅಸ್ಥಿರ("" ::: "memory");
            vint8m4_t q8v = __riscv_vle8_v_i8m4(q8, 64);
            q8 += 64;

            vint16m8_t prod = __riscv_vwmul_vv_i16m8(q2_final, q8v, 64);
            asm ಅಸ್ಥಿರ("" ::: "memory");
            vint32m1_t zero_vec = __riscv_vmv_v_x_i32m1(0, 1);

            int32_t sum0 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(
                           __riscv_vget_v_i16m8_i16m2(prod, 0), zero_vec, 16));

            int32_t sum1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(
                           __riscv_vget_v_i16m8_i16m2(prod, 1), zero_vec, 16));

            int32_t sum2 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(
                           __riscv_vget_v_i16m8_i16m2(prod, 2), zero_vec, 16));

            int32_t sum3 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(
                           __riscv_vget_v_i16m8_i16m2(prod, 3), zero_vec, 16));

            ಸ್ಥಿರಾಂಕ uint8_t scale_byte_1 = scales[0];
            ಸ್ಥಿರಾಂಕ uint8_t scale_byte_2 = scales[1];
            scales += 2;

            ಮೊತ್ತ_int += sum0 * ((scale_byte_1 & 0x0F) * 2 + 1);
            ಮೊತ್ತ_int += sum1 * ((scale_byte_1 >> 4)   * 2 + 1);
            ಮೊತ್ತ_int += sum2 * ((scale_byte_2 & 0x0F) * 2 + 1);
            ಮೊತ್ತ_int += sum3 * ((scale_byte_2 >> 4)   * 2 + 1);
        }

        sumf += d * ಮೊತ್ತ_int;
    }
    *s = 0.125f * sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq2_xs_q8_K_vl256(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_xs * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K   * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;
    ಸ್ಥಿರಾಂಕ uint64_t * signs64 = (ಸ್ಥಿರಾಂಕ uint64_t *)keven_signs_q2xs;
    ಸ್ಥಿರಾಂಕ uint64_t * grid64  = (ಸ್ಥಿರಾಂಕ uint64_t *)iq2xs_grid;

    ದಶಮಾಂಶ sumf = 0.0f;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d;
        ಸ್ಥಿರಾಂಕ uint16_t * GGML_RESTRICT qs = x[i].qs;
        ಸ್ಥಿರಾಂಕ int8_t   * GGML_RESTRICT q8 = y[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t  * GGML_RESTRICT scales = x[i].scales;

        int32_t ಮೊತ್ತ_int = 0;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ib128 = 0; ib128 < 2; ++ib128) {

            vuint16m1_t v_qs = __riscv_vle16_v_u16m1(qs, 16);
            qs += 16;

            // Prepare offsets for grid and signs
            vuint16m1_t vidx_grid = __riscv_vsll_vx_u16m1(__riscv_vand_vx_u16m1(v_qs, 511, 16), 3, 16);
            vuint16m1_t vidx_sign = __riscv_vsll_vx_u16m1(__riscv_vsrl_vx_u16m1(v_qs, 9, 16), 3, 16);

            // Indexed load 128 weights (16 x 8-byte chunks)
            vuint64m4_t vq2_64 = __riscv_vluxei16_v_u64m4(grid64, vidx_grid, 16);
            vuint64m4_t vs2_64 = __riscv_vluxei16_v_u64m4(signs64, vidx_sign, 16);

            vint8m4_t q2u = __riscv_vreinterpret_v_u8m4_i8m4(__riscv_vreinterpret_v_u64m4_u8m4(vq2_64));
            vint8m4_t q2s = __riscv_vreinterpret_v_u8m4_i8m4(__riscv_vreinterpret_v_u64m4_u8m4(vs2_64));

            // Apply signs to get dequantized IQ2 values
            vint8m4_t q2_final = __riscv_vmul_vv_i8m4(q2u, q2s, 128);
            asm ಅಸ್ಥಿರ("" ::: "memory");

            // Load corresponding Q8 weights
            vint8m4_t q8v = __riscv_vle8_v_i8m4(q8, 128);
            q8 += 128;

            vint16m8_t prod = __riscv_vwmul_vv_i16m8(q2_final, q8v, 128);
            asm ಅಸ್ಥಿರ("" ::: "memory");

            uint8_t sc0 = scales[0];
            uint8_t sc1 = scales[1];
            uint8_t sc2 = scales[2];
            uint8_t sc3 = scales[3];
            scales += 4;

            vint32m1_t zero_vec = __riscv_vmv_v_x_i32m1(0, 1);

            // 9. Reduce each 16-element chunk and apply corresponding nibble scale

            int32_t s0 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(prod, 0), zero_vec, 16));
            ಮೊತ್ತ_int += s0 * ((sc0 & 0x0F) * 2 + 1);

            int32_t s1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(prod, 1), zero_vec, 16));
            ಮೊತ್ತ_int += s1 * ((sc0 >> 4) * 2 + 1);

            int32_t s2 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(prod, 2), zero_vec, 16));
            ಮೊತ್ತ_int += s2 * ((sc1 & 0x0F) * 2 + 1);

            int32_t s3 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(prod, 3), zero_vec, 16));
            ಮೊತ್ತ_int += s3 * ((sc1 >> 4) * 2 + 1);

            int32_t s4 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(prod, 4), zero_vec, 16));
            ಮೊತ್ತ_int += s4 * ((sc2 & 0x0F) * 2 + 1);

            int32_t s5 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(prod, 5), zero_vec, 16));
            ಮೊತ್ತ_int += s5 * ((sc2 >> 4) * 2 + 1);

            int32_t s6 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(prod, 6), zero_vec, 16));
            ಮೊತ್ತ_int += s6 * ((sc3 & 0x0F) * 2 + 1);

            int32_t s7 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(prod, 7), zero_vec, 16));
            ಮೊತ್ತ_int += s7 * ((sc3 >> 4) * 2 + 1);
        }

        sumf += d * (ದಶಮಾಂಶ)ಮೊತ್ತ_int;
    }
    *s = 0.125f * sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq2_xs_q8_K_vl512(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc); ಬಳಸದ(bx); ಬಳಸದ(by); ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_xs * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K   * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;
    ಸ್ಥಿರಾಂಕ uint64_t * signs64 = (ಸ್ಥಿರಾಂಕ uint64_t *)keven_signs_q2xs;
    ಸ್ಥಿರಾಂಕ uint64_t * grid64  = (ಸ್ಥಿರಾಂಕ uint64_t *)iq2xs_grid;

    ದಶಮಾಂಶ sumf = 0.0f;
     ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ combined_scale = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d;

        ಸ್ಥಿರಾಂಕ uint16_t * GGML_RESTRICT qs = x[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t  * GGML_RESTRICT scales = x[i].scales;
        ಸ್ಥಿರಾಂಕ int8_t   * GGML_RESTRICT q8 = y[i].qs;

        vint8m4_t q8_all = __riscv_vle8_v_i8m4(q8, 256);

        // Load indices ---
        vuint16m1_t v_qs = __riscv_vle16_v_u16m1(qs, 32);

        // Extract low 9 bits and multiply by 8 (shift left 3) for byte offset into uint64 table
        vuint16m1_t vidx_grid = __riscv_vsll_vx_u16m1(__riscv_vand_vx_u16m1(v_qs, 511, 32), 3, 32);

        // Extract high 7 bits (shift right 9) and multiply by 8 (shift left 3) for byte offset
        vuint16m1_t vidx_sign = __riscv_vsll_vx_u16m1(__riscv_vsrl_vx_u16m1(v_qs, 9, 32), 3, 32);

        vuint64m4_t vq2_64 = __riscv_vluxei16_v_u64m4(grid64, vidx_grid, 32);
        vuint64m4_t vs2_64 = __riscv_vluxei16_v_u64m4(signs64, vidx_sign, 32);

        vint8m4_t q2_all = __riscv_vreinterpret_v_u8m4_i8m4(__riscv_vreinterpret_v_u64m4_u8m4(vq2_64));
        vint8m4_t s2_all = __riscv_vreinterpret_v_u8m4_i8m4(__riscv_vreinterpret_v_u64m4_u8m4(vs2_64));

        vint8m4_t q2_signed = __riscv_vmul_vv_i8m4(q2_all, s2_all, 256);
        vint16m8_t dot_all = __riscv_vwmul_vv_i16m8(q2_signed, q8_all, 256);
        ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;
        vint32m1_t zero_vec = __riscv_vmv_v_x_i32m1(0, 1);

#ಸೂಚನೆ GCC unroll 1
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; ++j) {
            uint8_t sc = scales[j];
            int16_t sc_lo = 2 * (sc & 0x0F) + 1;
            int16_t sc_hi = 2 * (sc >> 4)   + 1;

            vint32m1_t ಮೊತ್ತ_v0 = __riscv_vwredsum_vs_i16m8_i32m1(
                __riscv_vslidedown_vx_i16m8(dot_all, j * 32, 16), zero_vec, 16);
            int32_t isum0 = __riscv_vmv_x_s_i32m1_i32(ಮೊತ್ತ_v0);

            vint32m1_t ಮೊತ್ತ_v1 = __riscv_vwredsum_vs_i16m8_i32m1(
                __riscv_vslidedown_vx_i16m8(dot_all, j * 32 + 16, 16), zero_vec, 16);
            int32_t isum1 = __riscv_vmv_x_s_i32m1_i32(ಮೊತ್ತ_v1);

            ಮೊತ್ತ += (ದಶಮಾಂಶ)isum0 * sc_lo + (ದಶಮಾಂಶ)isum1 * sc_hi;
        }

        sumf += ಮೊತ್ತ * combined_scale;
    }
    *s = 0.125f * sumf;
}
#ಆದರೆ_ಅಂತ್ಯ

ಶೂನ್ಯ ggml_vec_dot_iq2_xs_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
#ಆದರೆ defined __riscv_v
      ಆಯ್ಕೆ (__riscv_vlenb() * 8) {
          ಸ್ಥಿತಿ 128:
              ggml_vec_dot_iq2_xs_q8_K_vl128(n, s, bs, vx, bx, vy, by, nrc);
              ನಿಲ್ಲಿಸು;
          ಸ್ಥಿತಿ 256:
              ggml_vec_dot_iq2_xs_q8_K_vl256(n, s, bs, vx, bx, vy, by, nrc);
              ನಿಲ್ಲಿಸು;
          ಪೂರ್ವನಿಯೋಜಿತ: // 512 and above
              ggml_vec_dot_iq2_xs_q8_K_vl512(n, s, bs, vx, bx, vy, by, nrc);
              ನಿಲ್ಲಿಸು;
      }
#ಇಲ್ಲದಿದ್ದರೆ
    ggml_vec_dot_iq2_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

#ಆದರೆ defined __riscv_v
ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq2_xxs_q8_K_vl128(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_xxs * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K    * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;
    ಸ್ಥಿರಾಂಕ uint64_t * signs64 = (ಸ್ಥಿರಾಂಕ uint64_t *)keven_signs_q2xs;
    ಸ್ಥಿರಾಂಕ uint64_t * grid64  = (ಸ್ಥಿರಾಂಕ uint64_t *)iq2xxs_grid;

    uint32_t ಸರಿಸು_constants[4] = {0, 7, 14, 21};
    vuint32m1_t v_shifts = __riscv_vle32_v_u32m1(ಸರಿಸು_constants, 4);

    ದಶಮಾಂಶ sumf = 0.0f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ combined_scale = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d;

        ಸ್ಥಿರಾಂಕ uint8_t  * GGML_RESTRICT q2_ptr = (ಸ್ಥಿರಾಂಕ uint8_t *) x[i].qs;
        ಸ್ಥಿರಾಂಕ int8_t   * GGML_RESTRICT q8 = y[i].qs;

        ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;

        #ಸೂಚನೆ GCC unroll 1
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            vint8m2_t q8_1 = __riscv_vle8_v_i8m2(q8, 32); q8 += 32;
            vint8m2_t q8_2 = __riscv_vle8_v_i8m2(q8, 32); q8 += 32;

            vuint8mf4_t v_raw_q2_1 = __riscv_vle8_v_u8mf4(q2_ptr, 4);
            vuint8mf4_t v_raw_q2_2 = __riscv_vle8_v_u8mf4(q2_ptr + 8, 4);

            vuint16mf2_t vidx_q2_1 = __riscv_vwcvtu_x_x_v_u16mf2(v_raw_q2_1, 4);
            vuint16mf2_t vidx_q2_2 = __riscv_vwcvtu_x_x_v_u16mf2(v_raw_q2_2, 4);

            vidx_q2_1 = __riscv_vsll_vx_u16mf2(vidx_q2_1, 3, 4);
            vidx_q2_2 = __riscv_vsll_vx_u16mf2(vidx_q2_2, 3, 4);

            uint32_t s_packed_1, s_packed_2;
            ಸ್ಮರಣೆ_ನಕಲು(&s_packed_1, q2_ptr + 4, 4);
            ಸ್ಮರಣೆ_ನಕಲು(&s_packed_2, q2_ptr + 12, 4);

            vuint32m1_t v_s_1 = __riscv_vmv_v_x_u32m1(s_packed_1, 4);
            vuint32m1_t v_s_2 = __riscv_vmv_v_x_u32m1(s_packed_2, 4);
            v_s_1 = __riscv_vsrl_vv_u32m1(v_s_1, v_shifts, 4);
            v_s_2 = __riscv_vsrl_vv_u32m1(v_s_2, v_shifts, 4);

            v_s_1 = __riscv_vand_vx_u32m1(v_s_1, 127, 4);
            v_s_2 = __riscv_vand_vx_u32m1(v_s_2, 127, 4);

            vuint16mf2_t vidx_s2_1 = __riscv_vsll_vx_u16mf2(__riscv_vncvt_x_x_w_u16mf2(v_s_1, 4), 3, 4);
            vuint16mf2_t vidx_s2_2 = __riscv_vsll_vx_u16mf2(__riscv_vncvt_x_x_w_u16mf2(v_s_2, 4), 3, 4);

            vuint64m2_t vq2_64_1 = __riscv_vluxei16_v_u64m2(grid64, vidx_q2_1, 4);
            vuint64m2_t vq2_64_2 = __riscv_vluxei16_v_u64m2(grid64, vidx_q2_2, 4);

            vint8m2_t q2_1 = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vreinterpret_v_u64m2_u8m2(vq2_64_1));
            vint8m2_t q2_2 = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vreinterpret_v_u64m2_u8m2(vq2_64_2));

            vuint64m2_t vs2_64_1 = __riscv_vluxei16_v_u64m2(signs64, vidx_s2_1, 4);
            vuint64m2_t vs2_64_2 = __riscv_vluxei16_v_u64m2(signs64, vidx_s2_2, 4);
            vint8m2_t s2_1 = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vreinterpret_v_u64m2_u8m2(vs2_64_1));
            vint8m2_t s2_2 = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vreinterpret_v_u64m2_u8m2(vs2_64_2));

            vint8m2_t q8s_1 = __riscv_vmul_vv_i8m2(q8_1, s2_1, 32);
            vint8m2_t q8s_2 = __riscv_vmul_vv_i8m2(q8_2, s2_2, 32);

            vint16m4_t dot1 = __riscv_vwmul_vv_i16m4(q8s_1, q2_1, 32);
            vint16m4_t dot2 = __riscv_vwmul_vv_i16m4(q8s_2, q2_2, 32);

            vint32m1_t zero_vec = __riscv_vmv_v_x_i32m1(0, 1);
            vint32m1_t sumv1 = __riscv_vwredsum_vs_i16m4_i32m1(dot1, zero_vec, 32);
            vint32m1_t sumv2 = __riscv_vwredsum_vs_i16m4_i32m1(dot2, zero_vec, 32);

            int32_t scalar_sum1 = __riscv_vmv_x_s_i32m1_i32(sumv1);
            int32_t scalar_sum2 = __riscv_vmv_x_s_i32m1_i32(sumv2);

            int16_t scale1 = 2 * ((s_packed_1 >> 28) & 0xF) + 1;
            int16_t scale2 = 2 * ((s_packed_2 >> 28) & 0xF) + 1;

            ಮೊತ್ತ += scalar_sum1 * scale1 + scalar_sum2 * scale2;
            q2_ptr += 16;
        }
        sumf += ಮೊತ್ತ * combined_scale;
    }
    *s = 0.125f * sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq2_xxs_q8_K_vl256(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_xxs * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K    * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;
    ಸ್ಥಿರಾಂಕ uint64_t * signs64 = (ಸ್ಥಿರಾಂಕ uint64_t *)keven_signs_q2xs;
    ಸ್ಥಿರಾಂಕ uint64_t * grid64  = (ಸ್ಥಿರಾಂಕ uint64_t *)iq2xxs_grid;

    uint32_t ಸರಿಸು_constants[4] = {0, 7, 14, 21};
    vuint32mf2_t v_shifts = __riscv_vle32_v_u32mf2(ಸರಿಸು_constants, 4);

    ದಶಮಾಂಶ sumf = 0.0f;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ combined_scale = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d;

        ಸ್ಥಿರಾಂಕ uint8_t  * GGML_RESTRICT q2_ptr = (ಸ್ಥಿರಾಂಕ uint8_t *) x[i].qs;
        ಸ್ಥಿರಾಂಕ int8_t   * GGML_RESTRICT q8 = y[i].qs;

        ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            vint8m1_t q8_1 = __riscv_vle8_v_i8m1(q8, 32); q8 += 32;
            vint8m1_t q8_2 = __riscv_vle8_v_i8m1(q8, 32); q8 += 32;

            vuint8mf8_t v_raw_q2_1 = __riscv_vle8_v_u8mf8(q2_ptr, 4);
            vuint8mf8_t v_raw_q2_2 = __riscv_vle8_v_u8mf8(q2_ptr + 8, 4);

            vuint16mf4_t vidx_q2_1 = __riscv_vwcvtu_x_x_v_u16mf4(v_raw_q2_1, 4);
            vuint16mf4_t vidx_q2_2 = __riscv_vwcvtu_x_x_v_u16mf4(v_raw_q2_2, 4);

            vidx_q2_1 = __riscv_vsll_vx_u16mf4(vidx_q2_1, 3, 4);
            vidx_q2_2 = __riscv_vsll_vx_u16mf4(vidx_q2_2, 3, 4);

            uint32_t s_packed_1, s_packed_2;
            ಸ್ಮರಣೆ_ನಕಲು(&s_packed_1, q2_ptr + 4, 4);
            ಸ್ಮರಣೆ_ನಕಲು(&s_packed_2, q2_ptr + 12, 4);

            vuint32mf2_t v_s_1 = __riscv_vmv_v_x_u32mf2(s_packed_1, 4);
            vuint32mf2_t v_s_2 = __riscv_vmv_v_x_u32mf2(s_packed_2, 4);

            v_s_1 = __riscv_vsrl_vv_u32mf2(v_s_1, v_shifts, 4);
            v_s_2 = __riscv_vsrl_vv_u32mf2(v_s_2, v_shifts, 4);

            v_s_1 = __riscv_vand_vx_u32mf2(v_s_1, 127, 4);
            v_s_2 = __riscv_vand_vx_u32mf2(v_s_2, 127, 4);

            // Narrow u32 -> u16 (vncvt) and Scale by 8 to get byte offsets
            vuint16mf4_t vidx_s2_1 = __riscv_vsll_vx_u16mf4(__riscv_vncvt_x_x_w_u16mf4(v_s_1, 4), 3, 4);
            vuint16mf4_t vidx_s2_2 = __riscv_vsll_vx_u16mf4(__riscv_vncvt_x_x_w_u16mf4(v_s_2, 4), 3, 4);

            // Load q2 values from lookup grid
            vuint64m1_t vq2_64_1 = __riscv_vluxei16_v_u64m1(grid64, vidx_q2_1, 4);
            vuint64m1_t vq2_64_2 = __riscv_vluxei16_v_u64m1(grid64, vidx_q2_2, 4);
            vint8m1_t q2_1 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vreinterpret_v_u64m1_u8m1(vq2_64_1));
            vint8m1_t q2_2 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vreinterpret_v_u64m1_u8m1(vq2_64_2));

            // Load sign values
            vuint64m1_t vs2_64_1 = __riscv_vluxei16_v_u64m1(signs64, vidx_s2_1, 4);
            vuint64m1_t vs2_64_2 = __riscv_vluxei16_v_u64m1(signs64, vidx_s2_2, 4);
            vint8m1_t s2_1 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vreinterpret_v_u64m1_u8m1(vs2_64_1));
            vint8m1_t s2_2 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vreinterpret_v_u64m1_u8m1(vs2_64_2));

            // Apply signs to q8
            vint8m1_t q8s_1 = __riscv_vmul_vv_i8m1(q8_1, s2_1, 32);
            vint8m1_t q8s_2 = __riscv_vmul_vv_i8m1(q8_2, s2_2, 32);

            // multiplying q2 with q8
            vint16m2_t dot1 = __riscv_vwmul_vv_i16m2(q8s_1, q2_1, 32);
            vint16m2_t dot2 = __riscv_vwmul_vv_i16m2(q8s_2, q2_2, 32);

            vint32m1_t zero_vec = __riscv_vmv_v_x_i32m1(0, 1);
            vint32m1_t sumv1 = __riscv_vwredsum_vs_i16m2_i32m1(dot1, zero_vec, 32);
            vint32m1_t sumv2 = __riscv_vwredsum_vs_i16m2_i32m1(dot2, zero_vec, 32);
            int32_t scalar_sum1 = __riscv_vmv_x_s_i32m1_i32(sumv1);
            int32_t scalar_sum2 = __riscv_vmv_x_s_i32m1_i32(sumv2);
            int16_t scale1 = 2 * ((s_packed_1 >> 28) & 0xF) + 1;
            int16_t scale2 = 2 * ((s_packed_2 >> 28) & 0xF) + 1;

            ಮೊತ್ತ += scalar_sum1 * scale1 + scalar_sum2 * scale2;
            q2_ptr += 16;
        }
        sumf += ಮೊತ್ತ * combined_scale;
    }
    *s = 0.125f * sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq2_xxs_q8_K_vl512(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc); ಬಳಸದ(bx); ಬಳಸದ(by); ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_xxs * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K    * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;
    ಸ್ಥಿರಾಂಕ uint64_t * signs64 = (ಸ್ಥಿರಾಂಕ uint64_t *)keven_signs_q2xs;
    ಸ್ಥಿರಾಂಕ uint64_t * grid64  = (ಸ್ಥಿರಾಂಕ uint64_t *)iq2xxs_grid;
    // Shift pattern {0,7,14,21} repeated 8 times for all 8 sub-blocks
    uint8_t ಸರಿಸು_arr[32] = {
        0, 7, 14, 21, 0, 7, 14, 21, 0, 7, 14, 21, 0, 7, 14, 21,
        0, 7, 14, 21, 0, 7, 14, 21, 0, 7, 14, 21, 0, 7, 14, 21
    };
    vuint8mf2_t v_shifts = __riscv_vle8_v_u8mf2(ಸರಿಸು_arr, 32);

    // Gather pattern to broadcast the 8 sub-block scales across the 32 lookup slots
    uint8_t gather_arr[32] = {
        0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3,
        4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7
    };
    vuint8mf2_t v_sign_gather_idx = __riscv_vle8_v_u8mf2(gather_arr, 32);

    ದಶಮಾಂಶ sumf = 0.0f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ combined_scale = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d;

        ಸ್ಥಿರಾಂಕ uint8_t  * GGML_RESTRICT q2_ptr = (ಸ್ಥಿರಾಂಕ uint8_t *) x[i].qs;
        ಸ್ಥಿರಾಂಕ int8_t   * GGML_RESTRICT q8 = y[i].qs;
        vint8m4_t q8_all = __riscv_vle8_v_i8m4(q8, 256);

        // De-interleave all 8 Index/Scale pairs for the 8x32-element sub-blocks
        vuint32mf2x2_t tuple = __riscv_vlseg2e32_v_u32mf2x2((ಸ್ಥಿರಾಂಕ uint32_t*)q2_ptr, 8);
        vuint32mf2_t v_ind32 = __riscv_vget_v_u32mf2x2_u32mf2(tuple, 0);
        vuint32mf2_t v_sc32  = __riscv_vget_v_u32mf2x2_u32mf2(tuple, 1);

        vuint8mf2_t v_raw_q2 = __riscv_vreinterpret_v_u32mf2_u8mf2(v_ind32);
        vuint16m1_t vidx_q2 = __riscv_vwcvtu_x_x_v_u16m1(v_raw_q2, 32);
        vidx_q2 = __riscv_vsll_vx_u16m1(vidx_q2, 3, 32);

        vuint32m2_t v_s = __riscv_vrgatherei16_vv_u32m2(__riscv_vlmul_ext_v_u32mf2_u32m2(v_sc32), __riscv_vwcvtu_x_x_v_u16m1(v_sign_gather_idx,32), 32);
        v_s = __riscv_vsrl_vv_u32m2(v_s, __riscv_vwcvtu_x_x_v_u32m2(__riscv_vwcvtu_x_x_v_u16m1(v_shifts,32),32), 32);
        v_s = __riscv_vand_vx_u32m2(v_s, 127, 32);
        vuint16m1_t vidx_s2 = __riscv_vsll_vx_u16m1(__riscv_vncvt_x_x_w_u16m1(v_s, 32), 3, 32);

        vuint64m4_t vq2_64 = __riscv_vluxei16_v_u64m4(grid64, vidx_q2, 32);
        vuint64m4_t vs2_64 = __riscv_vluxei16_v_u64m4(signs64, vidx_s2, 32);
        vint8m4_t q2_all = __riscv_vreinterpret_v_u8m4_i8m4(__riscv_vreinterpret_v_u64m4_u8m4(vq2_64));
        vint8m4_t s2_all = __riscv_vreinterpret_v_u8m4_i8m4(__riscv_vreinterpret_v_u64m4_u8m4(vs2_64));

        vint8m4_t q8s_all = __riscv_vmul_vv_i8m4(q8_all, s2_all, 256);
        vint16m8_t dot_all = __riscv_vwmul_vv_i16m8(q8s_all, q2_all, 256);

        ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;
        vint32m1_t zero_vec = __riscv_vmv_v_x_i32m1(0, 1);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; ++j) {
            uint32_t s_p = __riscv_vmv_x_s_u32mf2_u32(__riscv_vslidedown_vx_u32mf2(v_sc32, j, 8));
            int16_t sc = 2 * ((s_p >> 28) & 0xF) + 1;
            dot_all=__riscv_vslidedown_vx_i16m8(dot_all,j*32,32);
            vint32m1_t ಮೊತ್ತ_v = __riscv_vwredsum_vs_i16m8_i32m1(dot_all, zero_vec, 32);
            int32_t isum = __riscv_vmv_x_s_i32m1_i32(ಮೊತ್ತ_v);
            ಮೊತ್ತ += (ದಶಮಾಂಶ)isum * sc;
        }

        sumf += ಮೊತ್ತ * combined_scale;
    }
    *s = 0.125f * sumf;
}
#ಆದರೆ_ಅಂತ್ಯ

ಶೂನ್ಯ ggml_vec_dot_iq2_xxs_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
#ಆದರೆ defined __riscv_v
    ಆಯ್ಕೆ (__riscv_vlenb() * 8) {
        ಸ್ಥಿತಿ 128:
            ggml_vec_dot_iq2_xxs_q8_K_vl128(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 256:
            ggml_vec_dot_iq2_xxs_q8_K_vl256(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: // 512 and above
            ggml_vec_dot_iq2_xxs_q8_K_vl512(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ggml_vec_dot_iq2_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

#ಆದರೆ defined __riscv_v
ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq3_s_q8_K_vl128(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಬಳಸದ(nrc); ಬಳಸದ(bx); ಬಳಸದ(by); ಬಳಸದ(bs);
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_s * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K  * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;
    ಸ್ಥಿರಾಂಕ uint32_t * grid32 = (ಸ್ಥಿರಾಂಕ uint32_t *)iq3s_grid;

    vuint8mf2_t v_id_8  = __riscv_vid_v_u8mf2(8);
    vuint8m2_t  v_id_32 = __riscv_vid_v_u8m2(32);

    // Keeping these in a tight scope to hint they're only needed for the mask computation.
    vuint8m2_t v_sign_gather_indices, v_sign_masks;
    {
        vuint8m2_t v_shifts  = __riscv_vand_vx_u8m2(v_id_32, 7, 32);
        vuint8m2_t v_one_32  = __riscv_vmv_v_x_u8m2(1, 32);
        v_sign_gather_indices = __riscv_vsrl_vx_u8m2(v_id_32, 3, 32);
        v_sign_masks          = __riscv_vsll_vv_u8m2(v_one_32, v_shifts, 32);
    }

    ದಶಮಾಂಶ sumf = 0.0f;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d              = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ combined_scale = d * y[i].d;

        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT qs     = x[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT qh     = x[i].qh;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT scales = x[i].scales;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT signs  = x[i].signs;
        ಸ್ಥಿರಾಂಕ int8_t  * GGML_RESTRICT q8     = y[i].qs;

        ದಶಮಾಂಶ ಮೊತ್ತ_ಖಂಡ = 0.0f;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ib = 0; ib < 8; ++ib) {

            // Grid lookup
            vuint8m2_t v_grid_u8;
            {
                vuint8mf2_t v_qs_u8 = __riscv_vle8_v_u8mf2(qs, 8);
                qs += 8;

                uint8_t     qh_val   = *qh++;
                vuint8mf2_t v_qh_val = __riscv_vmv_v_x_u8mf2(qh_val, 8);
                v_qh_val = __riscv_vsrl_vv_u8mf2(v_qh_val, v_id_8, 8);
                v_qh_val = __riscv_vand_vx_u8mf2(v_qh_val, 1, 8);

                vuint16m1_t v_qs_u16 = __riscv_vwcvtu_x_x_v_u16m1(v_qs_u8, 8);
                v_qs_u16 = __riscv_vsll_vx_u16m1(v_qs_u16, 2, 8);

                vuint16m1_t v_qh_u16 = __riscv_vwcvtu_x_x_v_u16m1(v_qh_val, 8);
                v_qh_u16 = __riscv_vsll_vx_u16m1(v_qh_u16, 10, 8);

                vuint16m1_t v_grid_offsets = __riscv_vor_vv_u16m1(v_qs_u16, v_qh_u16, 8);

                vuint32m2_t v_grid_packed = __riscv_vluxei16_v_u32m2(grid32, v_grid_offsets, 8);
                v_grid_u8 = __riscv_vreinterpret_v_u32m2_u8m2(v_grid_packed);
            }
            __asm__ ಅಸ್ಥಿರ ("" ::: "memory");

            //Sign application and dot product
            int32_t s_val;
            {
                vuint8mf4_t v_signs_raw  = __riscv_vle8_v_u8mf4(signs, 4);
                signs += 4;

                vuint8m2_t v_signs_ಮೂಲ = __riscv_vlmul_ext_v_u8mf4_u8m2(v_signs_raw);
                vuint8m2_t v_signs_bcast  = __riscv_vrgather_vv_u8m2(v_signs_ಮೂಲ, v_sign_gather_indices, 32);
                vuint8m2_t v_sign_ಬಿಟ್ಟುಗಳು    = __riscv_vand_vv_u8m2(v_signs_bcast, v_sign_masks, 32);
                vbool4_t   m_negative     = __riscv_vmsne_vx_u8m2_b4(v_sign_ಬಿಟ್ಟುಗಳು, 0, 32);

                vint8m2_t v_q8        = __riscv_vle8_v_i8m2(q8, 32);
                q8 += 32;

                vint8m2_t  v_q8_signed = __riscv_vrsub_vx_i8m2_mu(m_negative, v_q8, v_q8, 0, 32);
                vint16m4_t v_dot       = __riscv_vwmulsu_vv_i16m4(v_q8_signed, v_grid_u8, 32);

                vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, 1);
                s_val = __riscv_vmv_x_s_i32m1_i32(
                    __riscv_vwredsum_vs_i16m4_i32m1(v_dot, v_zero, 32));
            }
            __asm__ ಅಸ್ಥಿರ ("" ::: "memory");
            {
                uint8_t sc_byte = scales[ib >> 1];
                ಪೂರ್ಣಾಂಕ sc_val = (ib & 1) ? (sc_byte >> 4) : (sc_byte & 0xF);
                sc_val = sc_val * 2 + 1;
                ಮೊತ್ತ_ಖಂಡ += (ದಶಮಾಂಶ)(s_val * sc_val);
            }
        }
        sumf += ಮೊತ್ತ_ಖಂಡ * combined_scale;
    }
    *s = sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq3_s_q8_K_vl256(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_s * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K  * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    ಸ್ಥಿರಾಂಕ uint64_t * grid64 = (ಸ್ಥಿರಾಂಕ uint64_t *)iq3s_grid;

    // --- Pre-load Constants ---
    ಸ್ಥಿರಾಂಕ uint16_t qh_ಬಿಟ್_shifts_arr[16] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    };
    vuint8m2_t v_sign_gather_indices = __riscv_vle8_v_u8m2(sign_gather_indices_arr, 64);
    vuint8m2_t v_sign_masks = __riscv_vle8_v_u8m2(sign_ಬಿಟ್_masks_arr, 64);
    vuint16m1_t v_qh_shifts = __riscv_vle16_v_u16m1(qh_ಬಿಟ್_shifts_arr, 16);

    ದಶಮಾಂಶ sumf = 0.0f;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ combined_scale = d * y[i].d;

        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT qs = x[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT qh = x[i].qh;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT scales = x[i].scales;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT signs = x[i].signs;
        ಸ್ಥಿರಾಂಕ int8_t  * GGML_RESTRICT q8 = y[i].qs;

        ದಶಮಾಂಶ ಮೊತ್ತ_ಖಂಡ = 0.0f;

        // Loop: Process 64 weights (16 mini-blocks of 4) per iteration
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ib = 0; ib < 4; ++ib) {

            vuint8mf2_t v_qs_u8 = __riscv_vle8_v_u8mf2(qs, 16);
            qs += 16;

            uint16_t qh_val;
            ಸ್ಮರಣೆ_ನಕಲು(&qh_val, qh, 2);
            qh += 2;

            vuint16m1_t v_qh_val = __riscv_vmv_v_x_u16m1(qh_val, 16);
            // Extract bits: (qh >> i) & 1
            v_qh_val = __riscv_vsrl_vv_u16m1(v_qh_val, v_qh_shifts, 16);
            v_qh_val = __riscv_vand_vx_u16m1(v_qh_val, 1, 16);

            vuint16m1_t v_qs_u16 = __riscv_vwcvtu_x_x_v_u16m1(v_qs_u8, 16);
            v_qs_u16 = __riscv_vsll_vx_u16m1(v_qs_u16, 2, 16);
            v_qh_val = __riscv_vsll_vx_u16m1(v_qh_val, 10, 16);
            vuint16m1_t v_grid_offsets = __riscv_vor_vv_u16m1(v_qs_u16, v_qh_val, 16);

            // Grid value is 4xuint8
            vuint32m2_t v_grid_packed = __riscv_vluxei16_v_u32m2((ಸ್ಥಿರಾಂಕ uint32_t *)grid64, v_grid_offsets, 16);
            vuint8m2_t v_grid_u8 = __riscv_vreinterpret_v_u32m2_u8m2(v_grid_packed);
            vuint8mf4_t v_signs_raw = __riscv_vle8_v_u8mf4(signs, 8);
            signs += 8;

            // Generate sign mask
            vuint8m2_t v_signs_ಮೂಲ = __riscv_vlmul_ext_v_u8mf4_u8m2(v_signs_raw);
            vuint8m2_t v_signs_bcast = __riscv_vrgather_vv_u8m2(v_signs_ಮೂಲ, v_sign_gather_indices, 64);
            vuint8m2_t v_sign_ಬಿಟ್ಟುಗಳು = __riscv_vand_vv_u8m2(v_signs_bcast, v_sign_masks, 64);
            vbool4_t m_negative = __riscv_vmsne_vx_u8m2_b4(v_sign_ಬಿಟ್ಟುಗಳು, 0, 64);

            vint8m2_t v_q8 = __riscv_vle8_v_i8m2(q8, 64);
            q8 += 64;

            // Apply Signs
            vint8m2_t v_q8_signed = __riscv_vrsub_vx_i8m2_mu(m_negative, v_q8, v_q8, 0, 64);
            vint16m4_t v_dot = __riscv_vwmulsu_vv_i16m4(v_q8_signed, v_grid_u8, 64);

            // Reduction
            vint16m2_t v_dot_lo = __riscv_vget_v_i16m4_i16m2(v_dot, 0);
            vint16m2_t v_dot_hi = __riscv_vget_v_i16m4_i16m2(v_dot, 1);
            vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, 1);

            int32_t s_lo = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(v_dot_lo, v_zero, 32));
            int32_t s_hi = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(v_dot_hi, v_zero, 32));

            // Apply sub-scales
            uint8_t sc_byte = *scales++;
            ಪೂರ್ಣಾಂಕ sc_lo = (sc_byte & 0xF) * 2 + 1;
            ಪೂರ್ಣಾಂಕ sc_hi = (sc_byte >> 4)  * 2 + 1;

            ಮೊತ್ತ_ಖಂಡ += s_lo * sc_lo + s_hi * sc_hi;
        }
        sumf += ಮೊತ್ತ_ಖಂಡ * combined_scale;
    }
    *s = sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq3_s_q8_K_vl512(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಬಳಸದ(nrc); ಬಳಸದ(bx); ಬಳಸದ(by); ಬಳಸದ(bs);
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_s * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K  * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;
    ಸ್ಥಿರಾಂಕ uint32_t * grid32 = (ಸ್ಥಿರಾಂಕ uint32_t *)iq3s_grid;

    // Generate Constants
    vuint8mf2_t v_id_32 = __riscv_vid_v_u8mf2(32);
    vuint8mf2_t v_qh_gather = __riscv_vsrl_vx_u8mf2(v_id_32, 3, 32);
    vuint8mf2_t v_qh_shifts = __riscv_vand_vx_u8mf2(v_id_32, 7, 32);
    vuint8m2_t v_id_128 = __riscv_vid_v_u8m2(128);
    vuint8m2_t v_sign_gather = __riscv_vsrl_vx_u8m2(v_id_128, 3, 128); // byte index
    vuint8m2_t v_sign_ಸರಿಸು_amts = __riscv_vand_vx_u8m2(v_id_128, 7, 128); // bit shift
    vuint8m2_t v_one_128 = __riscv_vmv_v_x_u8m2(1, 128);
    vuint8m2_t v_sign_masks = __riscv_vsll_vv_u8m2(v_one_128, v_sign_ಸರಿಸು_amts, 128);
    vuint8m2_t v_scale_indices = __riscv_vsrl_vx_u8m2(v_id_128, 5, 128);

    ದಶಮಾಂಶ sumf = 0.0f;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ combined_scale = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d;

        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT qs = x[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT qh = x[i].qh;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT scales = x[i].scales;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT signs = x[i].signs;
        ಸ್ಥಿರಾಂಕ int8_t  * GGML_RESTRICT q8 = y[i].qs;

        ದಶಮಾಂಶ ಮೊತ್ತ_ಖಂಡ = 0.0f;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ib = 0; ib < 2; ++ib) {
            vuint8mf2_t v_qs_u8 = __riscv_vle8_v_u8mf2(qs, 32);
            qs += 32;
            vuint8mf2_t v_qh_loaded = __riscv_vle8_v_u8mf2(qh, 4);
            qh += 4;
            vuint8mf2_t v_qh_expanded = __riscv_vrgather_vv_u8mf2(v_qh_loaded, v_qh_gather, 32);
            v_qh_expanded = __riscv_vsrl_vv_u8mf2(v_qh_expanded, v_qh_shifts, 32);
            v_qh_expanded = __riscv_vand_vx_u8mf2(v_qh_expanded, 1, 32);
            vuint16m1_t v_qs_u16 = __riscv_vwcvtu_x_x_v_u16m1(v_qs_u8, 32);
            v_qs_u16 = __riscv_vsll_vx_u16m1(v_qs_u16, 2, 32); // * 4

            vuint16m1_t v_qh_u16 = __riscv_vwcvtu_x_x_v_u16m1(v_qh_expanded, 32);
            v_qh_u16 = __riscv_vsll_vx_u16m1(v_qh_u16, 10, 32); // * 256 * 4

            vuint16m1_t v_grid_offsets = __riscv_vor_vv_u16m1(v_qs_u16, v_qh_u16, 32);
            vuint32m2_t v_grid_packed = __riscv_vluxei16_v_u32m2(grid32, v_grid_offsets, 32);
            vuint8m2_t v_grid_u8 = __riscv_vreinterpret_v_u32m2_u8m2(v_grid_packed);
            vuint8mf2_t v_signs_raw = __riscv_vle8_v_u8mf2(signs, 16);
            signs += 16;

            vuint8m2_t v_signs_ಮೂಲ = __riscv_vlmul_ext_v_u8mf2_u8m2(v_signs_raw);
            vuint8m2_t v_signs_bcast = __riscv_vrgather_vv_u8m2(v_signs_ಮೂಲ, v_sign_gather, 128);
            vuint8m2_t v_sign_ಬಿಟ್ಟುಗಳು = __riscv_vand_vv_u8m2(v_signs_bcast, v_sign_masks, 128);
            vbool4_t m_negative = __riscv_vmsne_vx_u8m2_b4(v_sign_ಬಿಟ್ಟುಗಳು, 0, 128);

            vint8m2_t v_q8 = __riscv_vle8_v_i8m2(q8, 128);
            q8 += 128;

            vint8m2_t v_q8_signed = __riscv_vrsub_vx_i8m2_mu(m_negative, v_q8, v_q8, 0, 128);
            vint16m4_t v_dot = __riscv_vwmulsu_vv_i16m4(v_q8_signed, v_grid_u8, 128);
            uint16_t sc_raw;
            ಸ್ಮರಣೆ_ನಕಲು(&sc_raw, scales, 2);
            scales += 2; // Advance 2 bytes

            uint8_t sc_unpacked[4];
            sc_unpacked[0] = (sc_raw & 0xF);
            sc_unpacked[1] = (sc_raw >> 4) & 0xF;
            sc_unpacked[2] = (sc_raw >> 8) & 0xF;
            sc_unpacked[3] = (sc_raw >> 12) & 0xF;

            vuint8mf2_t v_sc_4 = __riscv_vle8_v_u8mf2(sc_unpacked, 4);
            v_sc_4 = __riscv_vmul_vx_u8mf2(v_sc_4, 2, 4);
            v_sc_4 = __riscv_vadd_vx_u8mf2(v_sc_4, 1, 4);
            vuint8m2_t v_sc_4_expanded = __riscv_vlmul_ext_v_u8mf2_u8m2(v_sc_4);
            vuint8m2_t v_scales_bcast = __riscv_vrgather_vv_u8m2(v_sc_4_expanded, v_scale_indices, 128);
            vint16m4_t v_scales_i16 = __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vwcvtu_x_x_v_u16m4(v_scales_bcast, 128));
            vint32m8_t v_weighted_ಮೊತ್ತ = __riscv_vwmul_vv_i32m8(v_dot, v_scales_i16, 128);
            vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, 1);
            int32_t s_val = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m8_i32m1(v_weighted_ಮೊತ್ತ, v_zero, 128));

            ಮೊತ್ತ_ಖಂಡ += s_val;
        }
        sumf += ಮೊತ್ತ_ಖಂಡ * combined_scale;
    }
    *s = sumf;
}
#ಆದರೆ_ಅಂತ್ಯ

ಶೂನ್ಯ ggml_vec_dot_iq3_s_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
#ಆದರೆ defined __riscv_v
    ಆಯ್ಕೆ (__riscv_vlenb() * 8) {
        ಸ್ಥಿತಿ 128:
            ggml_vec_dot_iq3_s_q8_K_vl128(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 256:
            ggml_vec_dot_iq3_s_q8_K_vl256(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: // 512 and above
            ggml_vec_dot_iq3_s_q8_K_vl512(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ggml_vec_dot_iq3_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

#ಆದರೆ defined __riscv_v
ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq3_xxs_q8_K_vl128(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಬಳಸದ(nrc); ಬಳಸದ(bx); ಬಳಸದ(by); ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_xxs * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K    * GGML_RESTRICT y = vy;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    ಸ್ಥಿರಾಂಕ uint64_t * signs64 = (ಸ್ಥಿರಾಂಕ uint64_t *)keven_signs_q2xs;
    ಸ್ಥಿರಾಂಕ uint32_t * grid32  = (ಸ್ಥಿರಾಂಕ uint32_t *)iq3xxs_grid;

    // constants for unpacking logic
    ಸ್ಥಿರಾಂಕ uint32_t shifts_val[8] = {0, 7, 14, 21, 0, 7, 14, 21};
    vuint32m2_t v_shifts = __riscv_vle32_v_u32m2(shifts_val, 8);

    ಸ್ಥಿರಾಂಕ uint32_t gather_idx_val[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    vuint32m2_t v_gather_idx = __riscv_vle32_v_u32m2(gather_idx_val, 8);

    uint32_t aux32[2];
    ದಶಮಾಂಶ sumf = 0.0f;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d;

        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT q3_indices = x[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT metadata   = x[i].qs + QK_K/4;
        ಸ್ಥಿರಾಂಕ int8_t  * GGML_RESTRICT q8         = y[i].qs;

        ದಶಮಾಂಶ ಖಂಡ_ಮೊತ್ತ = 0.0f;

        // Process 64 weights per loop
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ib = 0; ib < QK_K / 64; ++ib) {

            // load of metadata via memcpy
            ಸ್ಮರಣೆ_ನಕಲು(aux32, metadata, 2 * ಗಾತ್ರ(uint32_t));
            metadata += 2 * ಗಾತ್ರ(uint32_t);

            vuint8m1_t v_q3_idx_u8 = __riscv_vle8_v_u8m1(q3_indices, 16);
            q3_indices += 16;

            vuint16m2_t v_q3_idx_u16 = __riscv_vwmulu_vx_u16m2(v_q3_idx_u8, 4, 16);

            vuint32m4_t v_q3_magnitudes_u32 = __riscv_vluxei16_v_u32m4(grid32, v_q3_idx_u16, 16);

            vint8m4_t v_q3_magnitudes = __riscv_vreinterpret_v_u8m4_i8m4(
                                        __riscv_vreinterpret_v_u32m4_u8m4(v_q3_magnitudes_u32));

            vuint32m2_t v_aux = __riscv_vle32_v_u32m2(aux32, 2);

            vuint32m2_t v_aux_expanded = __riscv_vrgather_vv_u32m2(v_aux, v_gather_idx, 8);

            vuint32m2_t v_s_vals_raw = __riscv_vand_vx_u32m2(
                                       __riscv_vsrl_vv_u32m2(v_aux_expanded, v_shifts, 8), 127, 8);

            vuint16m1_t sign_indices_byte_ಸ್ಥಾನಾಂತರ = __riscv_vsll_vx_u16m1(
                                                   __riscv_vncvt_x_x_w_u16m1(v_s_vals_raw, 8), 3, 8);

            vuint64m4_t v_s_vals_u64 = __riscv_vluxei16_v_u64m4(signs64, sign_indices_byte_ಸ್ಥಾನಾಂತರ, 8);

            vint8m4_t v_s_vals = __riscv_vreinterpret_v_u8m4_i8m4(
                                 __riscv_vreinterpret_v_u64m4_u8m4(v_s_vals_u64));

            vint8m4_t v_q3_signed = __riscv_vmul_vv_i8m4(v_q3_magnitudes, v_s_vals, 64);
            asm ಅಸ್ಥಿರ("" ::: "memory");
            vint8m4_t v_q8 = __riscv_vle8_v_i8m4(q8, 64);
            q8 += 64;

            vint16m8_t v_dot = __riscv_vwmul_vv_i16m8(v_q8, v_q3_signed, 64);

            asm ಅಸ್ಥಿರ("" ::: "memory");

            vint16m4_t v_dot_1 = __riscv_vget_v_i16m8_i16m4(v_dot, 0);
            vint16m4_t v_dot_2 = __riscv_vget_v_i16m8_i16m4(v_dot, 1);

            vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, 1);

            vint32m1_t v_ಮೊತ್ತ_1 = __riscv_vwredsum_vs_i16m4_i32m1(v_dot_1, v_zero, 32);
            vint32m1_t v_ಮೊತ್ತ_2 = __riscv_vwredsum_vs_i16m4_i32m1(v_dot_2, v_zero, 32);

            int32_t sum1_i = __riscv_vmv_x_s_i32m1_i32(v_ಮೊತ್ತ_1);
            int32_t sum2_i = __riscv_vmv_x_s_i32m1_i32(v_ಮೊತ್ತ_2);

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale1_f = (ದಶಮಾಂಶ)(2 * (aux32[0] >> 28) + 1);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale2_f = (ದಶಮಾಂಶ)(2 * (aux32[1] >> 28) + 1);

            ಖಂಡ_ಮೊತ್ತ += sum1_i * scale1_f + sum2_i * scale2_f;
        }

        sumf += d * ಖಂಡ_ಮೊತ್ತ;
    }
    *s = 0.25f * sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq3_xxs_q8_K_vl256(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_xxs * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K    * GGML_RESTRICT y = vy;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    ಸ್ಥಿರಾಂಕ uint64_t * signs64 = (ಸ್ಥಿರಾಂಕ uint64_t *)keven_signs_q2xs;
    ಸ್ಥಿರಾಂಕ uint32_t * grid32  = (ಸ್ಥಿರಾಂಕ uint32_t *)iq3xxs_grid;

    // constants for unpacking logic
    ಸ್ಥಿರಾಂಕ uint32_t shifts_val[8] = {0, 7, 14, 21, 0, 7, 14, 21};
    vuint32m1_t v_shifts = __riscv_vle32_v_u32m1(shifts_val, 8);

    ಸ್ಥಿರಾಂಕ uint32_t gather_idx_val[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    vuint32m1_t v_gather_idx = __riscv_vle32_v_u32m1(gather_idx_val, 8);

    uint32_t aux32[2];
    ದಶಮಾಂಶ sumf = 0.0f;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d;

        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT q3_indices = x[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT metadata   = x[i].qs + QK_K/4;
        ಸ್ಥಿರಾಂಕ int8_t  * GGML_RESTRICT q8         = y[i].qs;

        ದಶಮಾಂಶ ಖಂಡ_ಮೊತ್ತ = 0.0f;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ib = 0; ib < QK_K / 64; ++ib) {
            // Load q8 (64 bytes)
            vint8m2_t v_q8 = __riscv_vle8_v_i8m2(q8, 64);
            q8 += 64;

            // load of metadata via memcpy
            ಸ್ಮರಣೆ_ನಕಲು(aux32, metadata, 2 * ಗಾತ್ರ(uint32_t));
            metadata += 2 * ಗಾತ್ರ(uint32_t);

            // Load q3 indices and gather magnitudes
            vuint8mf2_t v_q3_idx_u8 = __riscv_vle8_v_u8mf2(q3_indices, 16);
            q3_indices += 16;

            vuint16m1_t v_q3_idx_u16 = __riscv_vwmulu_vx_u16m1(v_q3_idx_u8, 4, 16);
            vuint32m2_t v_q3_magnitudes_u32 = __riscv_vluxei16_v_u32m2(grid32, v_q3_idx_u16, 16);
            vint8m2_t v_q3_magnitudes = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vreinterpret_v_u32m2_u8m2(v_q3_magnitudes_u32));

            // --- Unpacking of Sign Indices ---

            // 1. Load the 2 auxiliary 32-bit integers into a vector
            vuint32m1_t v_aux = __riscv_vle32_v_u32m1(aux32, 2);

            // 2. Broadcast/Gather: replicate aux[0] to first 4 lanes, aux[1] to next 4 lanes
            vuint32m1_t v_aux_expanded = __riscv_vrgather_vv_u32m1(v_aux, v_gather_idx, 8);

            // 3. Apply Shifts and Mask: ((val >> shift) & 127)
            vuint32m1_t v_s_vals_raw = __riscv_vand_vx_u32m1(__riscv_vsrl_vv_u32m1(v_aux_expanded, v_shifts, 8), 127, 8);

            // 4. Narrow to u16 (required for vluxei index) and multiply by 8 (byte offset for u64 table)
            vuint16mf2_t sign_indices_byte_ಸ್ಥಾನಾಂತರ = __riscv_vsll_vx_u16mf2(__riscv_vncvt_x_x_w_u16mf2(v_s_vals_raw, 8), 3, 8);

            // 5. Gather Signs
            vuint64m2_t v_s_vals_u64 = __riscv_vluxei16_v_u64m2(signs64, sign_indices_byte_ಸ್ಥಾನಾಂತರ, 8);
            vint8m2_t v_s_vals = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vreinterpret_v_u64m2_u8m2(v_s_vals_u64));

            vint8m2_t v_q3_signed = __riscv_vmul_vv_i8m2(v_q3_magnitudes, v_s_vals, 64);
            vint16m4_t v_dot = __riscv_vwmul_vv_i16m4(v_q8, v_q3_signed, 64);

            vint16m2_t v_dot_1 = __riscv_vget_v_i16m4_i16m2(v_dot, 0);
            vint16m2_t v_dot_2 = __riscv_vget_v_i16m4_i16m2(v_dot, 1);

            vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, 1);
            vint32m1_t v_ಮೊತ್ತ_1 = __riscv_vwredsum_vs_i16m2_i32m1(v_dot_1, v_zero, 32);
            vint32m1_t v_ಮೊತ್ತ_2 = __riscv_vwredsum_vs_i16m2_i32m1(v_dot_2, v_zero, 32);

            int32_t sum1_i = __riscv_vmv_x_s_i32m1_i32(v_ಮೊತ್ತ_1);
            int32_t sum2_i = __riscv_vmv_x_s_i32m1_i32(v_ಮೊತ್ತ_2);

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale1_f = (ದಶಮಾಂಶ)(2 * (aux32[0] >> 28) + 1);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale2_f = (ದಶಮಾಂಶ)(2 * (aux32[1] >> 28) + 1);

            ಖಂಡ_ಮೊತ್ತ += sum1_i * scale1_f + sum2_i * scale2_f;
        }

        sumf += d * ಖಂಡ_ಮೊತ್ತ;
    }
    *s = 0.25f * sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq3_xxs_q8_K_vl512(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc); ಬಳಸದ(bx); ಬಳಸದ(by); ಬಳಸದ(bs);
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_xxs * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K    * GGML_RESTRICT y = vy;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    ಸ್ಥಿರಾಂಕ uint64_t * signs64 = (ಸ್ಥಿರಾಂಕ uint64_t *)keven_signs_q2xs;
    ಸ್ಥಿರಾಂಕ uint32_t * grid32  = (ಸ್ಥಿರಾಂಕ uint32_t *)iq3xxs_grid;

    // generate constants for unpacking metadata words into sign indices
    vuint32m1_t v_shifts;
    {
        vuint32m1_t v_ತಳ = __riscv_vid_v_u32m1(16);
        vuint32m1_t v_mod4 = __riscv_vand_vx_u32m1(v_ತಳ, 3, 16);
        v_shifts = __riscv_vmul_vx_u32m1(v_mod4, 7, 16);
    }

    vuint16mf2_t v_gather_idx;
    {
        vuint16mf2_t v_idx = __riscv_vid_v_u16mf2(16);
        v_gather_idx = __riscv_vsrl_vx_u16mf2(v_idx, 2, 16);
    }

    ದಶಮಾಂಶ sumf = 0.0f;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d;

        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT q3_indices = x[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT metadata   = x[i].qs + QK_K/4;
        ಸ್ಥಿರಾಂಕ int8_t  * GGML_RESTRICT q8         = y[i].qs;

        ದಶಮಾಂಶ ಖಂಡ_ಮೊತ್ತ = 0.0f;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ib128 = 0; ib128 < 2; ++ib128) {

            vint8m2_t v_q8 = __riscv_vle8_v_i8m2(q8, 128);
            q8 += 128;
            vuint8mf2_t v_q3_idx_u8 = __riscv_vle8_v_u8mf2(q3_indices, 32);
            q3_indices += 32;

            vuint16m1_t v_q3_idx_u16 = __riscv_vwmulu_vx_u16m1(v_q3_idx_u8, 4, 32);
            vuint32m2_t v_q3_mag_u32 = __riscv_vluxei16_v_u32m2(grid32, v_q3_idx_u16, 32);
            vint8m2_t v_q3_magnitudes = __riscv_vreinterpret_v_u8m2_i8m2(
            __riscv_vreinterpret_v_u32m2_u8m2(v_q3_mag_u32));
            vuint32m1_t v_aux = __riscv_vreinterpret_v_u8m1_u32m1(__riscv_vle8_v_u8m1(metadata, 16));
            metadata += 4 * ಗಾತ್ರ(uint32_t);

            vuint32m1_t v_aux_expanded = __riscv_vrgatherei16_vv_u32m1(v_aux, v_gather_idx, 16);

            vuint32m1_t v_s_raw = __riscv_vand_vx_u32m1(
                __riscv_vsrl_vv_u32m1(v_aux_expanded, v_shifts, 16), 127, 16);
            vuint16mf2_t sign_byte_ಸ್ಥಾನಾಂತರ = __riscv_vsll_vx_u16mf2(
                __riscv_vncvt_x_x_w_u16mf2(v_s_raw, 16), 3, 16);
            vuint64m2_t v_s_u64 = __riscv_vluxei16_v_u64m2(signs64, sign_byte_ಸ್ಥಾನಾಂತರ, 16);
            vint8m2_t v_signs = __riscv_vreinterpret_v_u8m2_i8m2(
                __riscv_vreinterpret_v_u64m2_u8m2(v_s_u64));
            vint8m2_t v_q3_signed = __riscv_vmul_vv_i8m2(v_q3_magnitudes, v_signs, 128);
            vint16m4_t prod = __riscv_vwmul_vv_i16m4(v_q3_signed, v_q8, 128);

            vint32m1_t zero_vec = __riscv_vmv_v_x_i32m1(0, 1);
            int32_t group0_ಮೊತ್ತ = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(
                         __riscv_vget_v_i16m4_i16m1(prod, 0), zero_vec, 32));
            int32_t group1_ಮೊತ್ತ = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(
                         __riscv_vget_v_i16m4_i16m1(prod, 1), zero_vec, 32));
            int32_t group2_ಮೊತ್ತ = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(
                         __riscv_vget_v_i16m4_i16m1(prod, 2), zero_vec, 32));
            int32_t group3_ಮೊತ್ತ = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(
                         __riscv_vget_v_i16m4_i16m1(prod, 3), zero_vec, 32));

            vuint32m1_t v_scales_raw = __riscv_vsrl_vx_u32m1(v_aux, 28, 4);
            vuint32m1_t v_scales = __riscv_vadd_vx_u32m1(
                                        __riscv_vsll_vx_u32m1(v_scales_raw, 1, 4),
                                        1, 4);
            int32_t scale0 = (int32_t)__riscv_vmv_x_s_u32m1_u32(v_scales);
            int32_t scale1 = (int32_t)__riscv_vmv_x_s_u32m1_u32(__riscv_vslidedown_vx_u32m1(v_scales, 1, 4));
            int32_t scale2 = (int32_t)__riscv_vmv_x_s_u32m1_u32(__riscv_vslidedown_vx_u32m1(v_scales, 2, 4));
            int32_t scale3 = (int32_t)__riscv_vmv_x_s_u32m1_u32(__riscv_vslidedown_vx_u32m1(v_scales, 3, 4));

            ಖಂಡ_ಮೊತ್ತ += (ದಶಮಾಂಶ)(group0_ಮೊತ್ತ * scale0 + group1_ಮೊತ್ತ * scale1 +
                                 group2_ಮೊತ್ತ * scale2 + group3_ಮೊತ್ತ * scale3);
        }

        sumf += d * ಖಂಡ_ಮೊತ್ತ;
    }
    *s = 0.25f * sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq3_xxs_q8_K_vl1024(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc); ಬಳಸದ(bx); ಬಳಸದ(by); ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_xxs * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K    * GGML_RESTRICT y = vy;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    ಸ್ಥಿರಾಂಕ uint64_t * signs64 = (ಸ್ಥಿರಾಂಕ uint64_t *)keven_signs_q2xs;
    ಸ್ಥಿರಾಂಕ uint32_t * grid32  = (ಸ್ಥಿರಾಂಕ uint32_t *)iq3xxs_grid;

    vuint32m1_t v_shifts;
    {
        vuint32m1_t v_id   = __riscv_vid_v_u32m1(32);
        vuint32m1_t v_mod4 = __riscv_vand_vx_u32m1(v_id, 3, 32);
        v_shifts           = __riscv_vmul_vx_u32m1(v_mod4, 7, 32);
    }
    vuint16mf2_t v_gather_idx;
    {
        vuint16mf2_t v_id_16 = __riscv_vid_v_u16mf2(32);
        v_gather_idx         = __riscv_vsrl_vx_u16mf2(v_id_16, 2, 32);
    }

    ದಶಮಾಂಶ sumf = 0.0f;
    uint32_t aux32[8]; // Buffer for block metadata

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d) * y[i].d;

        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT q3_indices = x[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT metadata   = x[i].qs + QK_K/4;
        ಸ್ಥಿರಾಂಕ int8_t  * GGML_RESTRICT q8         = y[i].qs;

        vint8m2_t v_q8 = __riscv_vle8_v_i8m2(q8, 256);
        vuint8mf2_t v_q3_idx_raw = __riscv_vle8_v_u8mf2(q3_indices, 64);
        vuint16m1_t v_q3_idx_u16 = __riscv_vwmulu_vx_u16m1(v_q3_idx_raw, 4, 64);

        vuint32m2_t v_q3_grid_vals = __riscv_vluxei16_v_u32m2(grid32, v_q3_idx_u16, 64);

        vint8m2_t v_q3_mags = __riscv_vreinterpret_v_u8m2_i8m2(
                              __riscv_vreinterpret_v_u32m2_u8m2(v_q3_grid_vals));

        ಸ್ಮರಣೆ_ನಕಲು(aux32, metadata, 8 * ಗಾತ್ರ(uint32_t));
        vuint32m1_t v_aux_8 = __riscv_vle32_v_u32m1(aux32, 8);

        vuint32m1_t v_aux_32 = __riscv_vrgatherei16_vv_u32m1(v_aux_8, v_gather_idx, 32);

        vuint32m1_t v_sign_idx_raw = __riscv_vand_vx_u32m1(
                                     __riscv_vsrl_vv_u32m1(v_aux_32, v_shifts, 32), 127, 32);

        vuint16mf2_t v_sign_offsets = __riscv_vsll_vx_u16mf2(
                                      __riscv_vncvt_x_x_w_u16mf2(v_sign_idx_raw, 32), 3, 32);

        vuint64m2_t v_signs_u64 = __riscv_vluxei16_v_u64m2(signs64, v_sign_offsets, 32);

        vint8m2_t v_signs = __riscv_vreinterpret_v_u8m2_i8m2(
                            __riscv_vreinterpret_v_u64m2_u8m2(v_signs_u64));

        vint8m2_t v_q3_final = __riscv_vmul_vv_i8m2(v_q3_mags, v_signs, 256);

        vint16m4_t v_dot = __riscv_vwmul_vv_i16m4(v_q8, v_q3_final, 256);
        ದಶಮಾಂಶ ಖಂಡ_ಮೊತ್ತ = 0.0f;
        vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, 1);
        vint16m4_t v_accum = v_dot;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; ++j) {
            ದಶಮಾಂಶ scale = (ದಶಮಾಂಶ)(2 * (aux32[j] >> 28) + 1);

            vint32m1_t v_partial_ಮೊತ್ತ = __riscv_vwredsum_vs_i16m4_i32m1(v_accum, v_zero, 32);

            int32_t partial_ಮೊತ್ತ_i = __riscv_vmv_x_s_i32m1_i32(v_partial_ಮೊತ್ತ);
            ಖಂಡ_ಮೊತ್ತ += partial_ಮೊತ್ತ_i * scale;
            v_accum = __riscv_vslidedown_vx_i16m4(v_accum, 32, 32);

        }

        sumf += d * ಖಂಡ_ಮೊತ್ತ;
    }
    *s = 0.25f * sumf;
}
#ಆದರೆ_ಅಂತ್ಯ

ಶೂನ್ಯ ggml_vec_dot_iq3_xxs_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
#ಆದರೆ defined __riscv_v
    ಆಯ್ಕೆ (__riscv_vlenb() * 8) {
        ಸ್ಥಿತಿ 128:
            ggml_vec_dot_iq3_xxs_q8_K_vl128(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 256:
            ggml_vec_dot_iq3_xxs_q8_K_vl256(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 512:
            ggml_vec_dot_iq3_xxs_q8_K_vl512(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: // 1024 and above
            ggml_vec_dot_iq3_xxs_q8_K_vl1024(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ggml_vec_dot_iq3_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

#ಆದರೆ defined __riscv_v
ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq4_nl_q8_0_vl128(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);
    ಪ್ರತಿಜ್ಞಾ(n % QK4_NL == 0);
    static_assert(QK4_NL == QK8_0, "QK4_NL and QK8_0 must be the same");

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_nl * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0   * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK4_NL;

    ಪೂರ್ಣಾಂಕ ib = 0;
    ದಶಮಾಂಶ sumf = 0;

    // Load the lookup table once.
    ಸ್ಥಿರಾಂಕ vint8m2_t ಮೌಲ್ಯಗಳು = __riscv_vle8_v_i8m2(kvalues_iq4nl, 16);
    ಪೂರ್ಣಾಂಕ acc1, acc2;

    // We process 2 blocks at once.
    ಗಾಗಿ (; ib + 1 < nb; ib += 2) {
        // Weights and activations.
        vuint8m1_t iq4_packed1 = __riscv_vle8_v_u8m1(x[ib + 0].qs, 16);
        vint8m2_t q8b1 = __riscv_vle8_v_i8m2(y[ib + 0].qs, 32);
        vuint8m1_t iq4_packed2 = __riscv_vle8_v_u8m1(x[ib + 1].qs, 16);
        vint8m2_t q8b2 = __riscv_vle8_v_i8m2(y[ib + 1].qs, 32);

        // Unpack the weight blocks.
        vuint8m2_t iq4bits1 = __riscv_vcreate_v_u8m1_u8m2(
            __riscv_vand_vx_u8m1(iq4_packed1, 0xf, 16),
            __riscv_vsrl_vx_u8m1(iq4_packed1, 4, 16)
        );
        vuint8m2_t iq4bits2 = __riscv_vcreate_v_u8m1_u8m2(
            __riscv_vand_vx_u8m1(iq4_packed2, 0xf, 16),
            __riscv_vsrl_vx_u8m1(iq4_packed2, 4, 16)
        );

        // Gather values from the lookup table.
        vint8m2_t iq4b1 = __riscv_vrgather_vv_i8m2(ಮೌಲ್ಯಗಳು, iq4bits1, 32);
        vint8m2_t iq4b2 = __riscv_vrgather_vv_i8m2(ಮೌಲ್ಯಗಳು, iq4bits2, 32);

        // Accumulation.
        vint16m4_t sum1 = __riscv_vwmul_vv_i16m4(q8b1, iq4b1, 32);
        vint16m4_t sum2 = __riscv_vwmul_vv_i16m4(q8b2, iq4b2, 32);
        __riscv_vse32_v_i32m1(&acc1,__riscv_vwredsum_vs_i16m4_i32m1(sum1, __riscv_vmv_v_x_i32m1(0, 1), 32), 1);
        __riscv_vse32_v_i32m1(&acc2,__riscv_vwredsum_vs_i16m4_i32m1(sum2, __riscv_vmv_v_x_i32m1(0, 1), 32), 1);
        sumf += ((GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[ib + 0].d) * GGML_CPU_FP16_ಗಮ್ಯ_FP32(y[ib + 0].d) * acc1));
        sumf += ((GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[ib + 1].d) * GGML_CPU_FP16_ಗಮ್ಯ_FP32(y[ib + 1].d) * acc2));
    }

    *s = sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq4_nl_q8_0_vl256(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);
    ಪ್ರತಿಜ್ಞಾ(n % QK4_NL == 0);
    static_assert(QK4_NL == QK8_0, "QK4_NL and QK8_0 must be the same");

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_nl * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0   * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK4_NL;

    ಪೂರ್ಣಾಂಕ ib = 0;
    ದಶಮಾಂಶ sumf = 0;

    // Load the lookup table once.
    ಸ್ಥಿರಾಂಕ vint8mf2_t ಮೌಲ್ಯಗಳು = __riscv_vle8_v_i8mf2(kvalues_iq4nl, 16);
    ಪೂರ್ಣಾಂಕ acc1, acc2;

    // We process 2 blocks at once.
    ಗಾಗಿ (; ib + 1 < nb; ib += 2) {
        // Weights and activations.
        vuint8mf2_t iq4_packed1 = __riscv_vle8_v_u8mf2(x[ib + 0].qs, 16);
        vint8mf2_t q8b_lo1 = __riscv_vle8_v_i8mf2(y[ib + 0].qs, 16);
        vint8mf2_t q8b_hi1 = __riscv_vle8_v_i8mf2(y[ib + 0].qs + 16, 16);
        vuint8mf2_t iq4_packed2 = __riscv_vle8_v_u8mf2(x[ib + 1].qs, 16);
        vint8mf2_t q8b_lo2 = __riscv_vle8_v_i8mf2(y[ib + 1].qs, 16);
        vint8mf2_t q8b_hi2 = __riscv_vle8_v_i8mf2(y[ib + 1].qs + 16, 16);

        // Unpack the weight blocks.
        vuint8mf2_t iq4bits_lo1 = __riscv_vand_vx_u8mf2(iq4_packed1, 0xf, 16);
        vuint8mf2_t iq4bits_hi1 = __riscv_vsrl_vx_u8mf2(iq4_packed1, 4, 16);
        vuint8mf2_t iq4bits_lo2 = __riscv_vand_vx_u8mf2(iq4_packed2, 0xf, 16);
        vuint8mf2_t iq4bits_hi2 = __riscv_vsrl_vx_u8mf2(iq4_packed2, 4, 16);

        // Gather values from the lookup table.
        vint8mf2_t iq4b_lo1 = __riscv_vrgather_vv_i8mf2(ಮೌಲ್ಯಗಳು, iq4bits_lo1, 16);
        vint8mf2_t iq4b_hi1 = __riscv_vrgather_vv_i8mf2(ಮೌಲ್ಯಗಳು, iq4bits_hi1, 16);
        vint8mf2_t iq4b_lo2 = __riscv_vrgather_vv_i8mf2(ಮೌಲ್ಯಗಳು, iq4bits_lo2, 16);
        vint8mf2_t iq4b_hi2 = __riscv_vrgather_vv_i8mf2(ಮೌಲ್ಯಗಳು, iq4bits_hi2, 16);

        // Accumulation.
        vint16m1_t sum1 = __riscv_vwmul_vv_i16m1(q8b_lo1, iq4b_lo1, 16);
        sum1 = __riscv_vwmacc_vv_i16m1(sum1, q8b_hi1, iq4b_hi1, 16);
        vint16m1_t sum2 = __riscv_vwmul_vv_i16m1(q8b_lo2, iq4b_lo2, 16);
        sum2 = __riscv_vwmacc_vv_i16m1(sum2, q8b_hi2, iq4b_hi2, 16);
        __riscv_vse32_v_i32m1(&acc1,__riscv_vwredsum_vs_i16m1_i32m1(sum1, __riscv_vmv_v_x_i32m1(0, 1), 16), 1);
        __riscv_vse32_v_i32m1(&acc2,__riscv_vwredsum_vs_i16m1_i32m1(sum2, __riscv_vmv_v_x_i32m1(0, 1), 16), 1);
        sumf += ((GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[ib + 0].d) * GGML_CPU_FP16_ಗಮ್ಯ_FP32(y[ib + 0].d) * acc1));
        sumf += ((GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[ib + 1].d) * GGML_CPU_FP16_ಗಮ್ಯ_FP32(y[ib + 1].d) * acc2));
    }

    *s = sumf;
}
#ಆದರೆ_ಅಂತ್ಯ

ಶೂನ್ಯ ggml_vec_dot_iq4_nl_q8_0(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
#ಆದರೆ defined __riscv_v
    ಆಯ್ಕೆ (__riscv_vlenb() * 8) {
        ಸ್ಥಿತಿ 128:
            ggml_vec_dot_iq4_nl_q8_0_vl128(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: // 256 and above
            ggml_vec_dot_iq4_nl_q8_0_vl256(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ggml_vec_dot_iq4_nl_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

#ಆದರೆ defined __riscv_v
ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq4_xs_q8_K_vl128(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_xs * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K   * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    ಸ್ಥಿರಾಂಕ vint8m4_t ಮೌಲ್ಯಗಳು = __riscv_vle8_v_i8m4(kvalues_iq4nl, 16);
    ದಶಮಾಂಶ sumf = 0;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ibl = 0; ibl < nb; ++ibl) {
        ಸ್ಥಿರಾಂಕ int8_t  * q8 = y[ibl].qs;
        ಸ್ಥಿರಾಂಕ uint8_t * iq4 = x[ibl].qs;
        uint16_t h = x[ibl].scales_h;

        // We process 2 sub-blocks together.
        ಪೂರ್ಣಾಂಕ sumi1 = 0, sumi2 = 0;
        #ಸೂಚನೆ GCC unroll 1
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ib = 0; ib < QK_K / 64; ++ib) {
            // Load the packed weights.
            ಸ್ಥಿರಾಂಕ vuint8m2_t iq4_packed = __riscv_vle8_v_u8m2(iq4, 32);
            iq4 += 32;

            // Unpack the weight blocks.
            ಸ್ಥಿರಾಂಕ vuint8m2_t iq4bits_lo = __riscv_vand_vx_u8m2(iq4_packed, 0xf, 32);
            ಸ್ಥಿರಾಂಕ vuint8m2_t iq4bits_hi = __riscv_vsrl_vx_u8m2(iq4_packed, 4, 32);
            ಸ್ಥಿರಾಂಕ vuint8m4_t iq4bits = __riscv_vcreate_v_u8m2_u8m4(iq4bits_lo, iq4bits_hi);
            ಸ್ಥಿರಾಂಕ vuint8m4_t iq4bits_reorder = __riscv_vcreate_v_u8m1_u8m4(
                __riscv_vmv_v_v_u8m1(__riscv_vget_v_u8m4_u8m1(iq4bits, 0), 16),
                __riscv_vmv_v_v_u8m1(__riscv_vget_v_u8m4_u8m1(iq4bits, 2), 16),
                __riscv_vmv_v_v_u8m1(__riscv_vget_v_u8m4_u8m1(iq4bits, 1), 16),
                __riscv_vmv_v_v_u8m1(__riscv_vget_v_u8m4_u8m1(iq4bits, 3), 16)
            );
            ಸ್ಥಿರಾಂಕ vint8m4_t iq4b = __riscv_vrgather_vv_i8m4(ಮೌಲ್ಯಗಳು, iq4bits_reorder, 64);

            // Multiply with activations.
            ಸ್ಥಿರಾಂಕ vint8m4_t q8b = __riscv_vle8_v_i8m4(q8, 64);
            q8 += 64;
            ಸ್ಥಿರಾಂಕ vint16m8_t prod = __riscv_vwmul_vv_i16m8(iq4b, q8b, 64);

            // Reduce separately.
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ acc0 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(prod, 0), __riscv_vmv_v_x_i32m1(0, 1), 32));
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ acc1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(prod, 1), __riscv_vmv_v_x_i32m1(0, 1), 32));

            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ls1 = ((x[ibl].scales_l[ib] & 0xf)  | ((h << 4) & 0x30)) - 32;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ls2 = ((x[ibl].scales_l[ib] >>  4)  | ((h << 2) & 0x30)) - 32;
            h >>= 4;

            sumi1 += acc0 * ls1;
            sumi2 += acc1 * ls2;

            __asm__ __volatile__("" ::: "memory");
        }

        sumf += GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[ibl].d) * y[ibl].d * (sumi1 + sumi2);
    }

    *s = sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq4_xs_q8_K_vl256(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_xs * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K   * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    ಸ್ಥಿರಾಂಕ vint8m4_t ಮೌಲ್ಯಗಳು = __riscv_vle8_v_i8m4(kvalues_iq4nl, 16);
    ದಶಮಾಂಶ sumf = 0;

    // Indices for re-ordering IQ4 data.
    uint16_t ಸೂಚ್ಯಂಕ[16] = {
        0, 1, 8, 9,
        2, 3, 10, 11,
        4, 5,12, 13,
        6, 7, 14, 15,
    };
    vuint16m1_t i_vec = __riscv_vle16_v_u16m1(ಸೂಚ್ಯಂಕ, 16);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ibl = 0; ibl < nb; ++ibl) {
        ಸ್ಥಿರಾಂಕ int8_t  * q8 = y[ibl].qs;
        ಸ್ಥಿರಾಂಕ uint8_t * iq4 = x[ibl].qs;
        uint16_t h = x[ibl].scales_h;

        ಪೂರ್ಣಾಂಕ sumi1 = 0, sumi2 = 0, sumi3 = 0, sumi4 = 0;

        #ಸೂಚನೆ GCC unroll 1
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ib = 0; ib < QK_K / 128; ++ib) {
            // Weights and activations.
            vuint8m2_t iq4_packed = __riscv_vle8_v_u8m2(iq4, 64);
            iq4 += 64;

            // Unpack the weight blocks.
            vuint8m2_t iq4bits_lo = __riscv_vand_vx_u8m2(iq4_packed, 0xf, 64);
            vuint8m2_t iq4bits_hi = __riscv_vsrl_vx_u8m2(iq4_packed, 4, 64);
            vuint8m4_t iq4bits = __riscv_vcreate_v_u8m2_u8m4(iq4bits_lo, iq4bits_hi);
            vuint8m4_t iq4bits_reorder = __riscv_vreinterpret_v_u64m4_u8m4(__riscv_vrgatherei16_vv_u64m4(__riscv_vreinterpret_v_u8m4_u64m4(iq4bits), i_vec, 16));
            vint8m4_t iq4b = __riscv_vrgather_vv_i8m4(ಮೌಲ್ಯಗಳು, iq4bits_reorder, 128);

            __asm__ __volatile__("" ::: "memory");

            // Multiply with activations.
            vint8m4_t q8b = __riscv_vle8_v_i8m4(q8, 128);
            vint16m8_t prod = __riscv_vwmul_vv_i16m8(iq4b, q8b, 128);
            q8 += 128;

            __asm__ __volatile__("" ::: "memory");

            // Reduce separately.
            ಪೂರ್ಣಾಂಕ acc0 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(prod, 0), __riscv_vmv_v_x_i32m1(0, 1), 32));
            ಪೂರ್ಣಾಂಕ acc1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(prod, 1), __riscv_vmv_v_x_i32m1(0, 1), 32));
            ಪೂರ್ಣಾಂಕ acc2 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(prod, 2), __riscv_vmv_v_x_i32m1(0, 1), 32));
            ಪೂರ್ಣಾಂಕ acc3 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(prod, 3), __riscv_vmv_v_x_i32m1(0, 1), 32));

            ಪೂರ್ಣಾಂಕ ls1 = ((x[ibl].scales_l[ib * 2 + 0] & 0xf)  | ((h << 4) & 0x30)) - 32;
            ಪೂರ್ಣಾಂಕ ls2 = ((x[ibl].scales_l[ib * 2 + 0] >>  4)  | ((h << 2) & 0x30)) - 32;
            ಪೂರ್ಣಾಂಕ ls3 = ((x[ibl].scales_l[ib * 2 + 1] &  0xf) | ((h << 0) & 0x30)) - 32;
            ಪೂರ್ಣಾಂಕ ls4 = ((x[ibl].scales_l[ib * 2 + 1] >>  4)  | ((h >> 2) & 0x30)) - 32;
            h >>= 8;

            sumi1 += acc0 * ls1;
            sumi2 += acc1 * ls2;
            sumi3 += acc2 * ls3;
            sumi4 += acc3 * ls4;

            __asm__ __volatile__("" ::: "memory");
        }

        sumf += GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[ibl].d) * y[ibl].d * (sumi1 + sumi2 + sumi3 + sumi4);
    }

    *s = sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq4_xs_q8_K_vl512(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_xs * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K   * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    ಸ್ಥಿರಾಂಕ vint8m4_t ಮೌಲ್ಯಗಳು = __riscv_vle8_v_i8m4(kvalues_iq4nl, 16);
    ದಶಮಾಂಶ sumf = 0;

    // Indices for re-ordering IQ4 data.
    ಸ್ಥಿರಾಂಕ uint16_t ಸೂಚ್ಯಂಕ[32] = {
        0, 1, 16, 17,
        2, 3, 18, 19,
        4, 5,20, 21,
        6, 7, 22, 23,
        8, 9, 24, 25,
        10, 11, 26, 27,
        12, 13,28, 29,
        14, 15, 30, 31,
    };
    ಸ್ಥಿರಾಂಕ vuint16m1_t i_vec = __riscv_vle16_v_u16m1(ಸೂಚ್ಯಂಕ, 32);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ibl = 0; ibl < nb; ++ibl) {
        ಸ್ಥಿರಾಂಕ int8_t  * q8 = y[ibl].qs;
        ಸ್ಥಿರಾಂಕ uint8_t * iq4 = x[ibl].qs;
        uint16_t h = x[ibl].scales_h;

        ಪೂರ್ಣಾಂಕ sumi = 0;

        #ಸೂಚನೆ GCC unroll 1
        // Process the entire super-block together.
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ib = 0; ib < QK_K / 256; ++ib) {
            // Weights and activations.
            ಸ್ಥಿರಾಂಕ vuint8m2_t iq4_packed = __riscv_vle8_v_u8m2(iq4, 128);
            iq4 += 128;

            // Unpack the weight blocks.
            ಸ್ಥಿರಾಂಕ vuint8m2_t iq4bits_lo = __riscv_vand_vx_u8m2(iq4_packed, 0xf, 128);
            ಸ್ಥಿರಾಂಕ vuint8m2_t iq4bits_hi = __riscv_vsrl_vx_u8m2(iq4_packed, 4, 128);
            ಸ್ಥಿರಾಂಕ vuint8m4_t iq4bits = __riscv_vcreate_v_u8m2_u8m4(iq4bits_lo, iq4bits_hi);
            ಸ್ಥಿರಾಂಕ vuint8m4_t iq4bits_reorder = __riscv_vreinterpret_v_u64m4_u8m4(__riscv_vrgatherei16_vv_u64m4(__riscv_vreinterpret_v_u8m4_u64m4(iq4bits), i_vec, 32));
            ಸ್ಥಿರಾಂಕ vint8m4_t iq4b = __riscv_vrgather_vv_i8m4(ಮೌಲ್ಯಗಳು, iq4bits_reorder, 256);

            __asm__ __volatile__("" ::: "memory");

            // Multiply with activations.
            ಸ್ಥಿರಾಂಕ vint8m4_t q8b = __riscv_vle8_v_i8m4(q8, 256);
            ಸ್ಥಿರಾಂಕ vint16m8_t prod = __riscv_vwmul_vv_i16m8(iq4b, q8b, 256);
            q8 += 256;

            // Reduce separately.
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ acc0 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(prod, 0), __riscv_vmv_v_x_i32m1(0, 1), 32));
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ acc1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(prod, 1), __riscv_vmv_v_x_i32m1(0, 1), 32));
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ acc2 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(prod, 2), __riscv_vmv_v_x_i32m1(0, 1), 32));
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ acc3 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(prod, 3), __riscv_vmv_v_x_i32m1(0, 1), 32));
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ acc4 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(prod, 4), __riscv_vmv_v_x_i32m1(0, 1), 32));
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ acc5 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(prod, 5), __riscv_vmv_v_x_i32m1(0, 1), 32));
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ acc6 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(prod, 6), __riscv_vmv_v_x_i32m1(0, 1), 32));
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ acc7 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(prod, 7), __riscv_vmv_v_x_i32m1(0, 1), 32));


            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ls0 = ((x[ibl].scales_l[0] & 0xf)  | ((h << 4) & 0x30)) - 32;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ls1 = ((x[ibl].scales_l[0] >>  4)  | ((h << 2) & 0x30)) - 32;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ls2 = ((x[ibl].scales_l[1] &  0xf) | ((h << 0) & 0x30)) - 32;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ls3 = ((x[ibl].scales_l[1] >>  4)  | ((h >> 2) & 0x30)) - 32;
            h >>= 8;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ls4 = ((x[ibl].scales_l[2] & 0xf)  | ((h << 4) & 0x30)) - 32;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ls5 = ((x[ibl].scales_l[2] >>  4)  | ((h << 2) & 0x30)) - 32;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ls6 = ((x[ibl].scales_l[3] &  0xf) | ((h << 0) & 0x30)) - 32;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ls7 = ((x[ibl].scales_l[3] >>  4)  | ((h >> 2) & 0x30)) - 32;

            sumi += acc0 * ls0;
            sumi += acc1 * ls1;
            sumi += acc2 * ls2;
            sumi += acc3 * ls3;
            sumi += acc4 * ls4;
            sumi += acc5 * ls5;
            sumi += acc6 * ls6;
            sumi += acc7 * ls7;

            __asm__ __volatile__("" ::: "memory");
        }

        sumf += GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[ibl].d) * y[ibl].d * (sumi);
    }

    *s = sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_iq4_xs_q8_K_vl1024(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_xs * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K   * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    ಸ್ಥಿರಾಂಕ vint8m2_t ಮೌಲ್ಯಗಳು = __riscv_vle8_v_i8m2(kvalues_iq4nl, 16);
    ದಶಮಾಂಶ sumf = 0;

    // Indices for re-ordering IQ4 data.
    ಸ್ಥಿರಾಂಕ uint16_t ಸೂಚ್ಯಂಕ[32] = {
        0, 1, 16, 17,
        2, 3, 18, 19,
        4, 5,20, 21,
        6, 7, 22, 23,
        8, 9, 24, 25,
        10, 11, 26, 27,
        12, 13,28, 29,
        14, 15, 30, 31,
    };
    ಸ್ಥಿರಾಂಕ vuint16mf2_t i_vec = __riscv_vle16_v_u16mf2(ಸೂಚ್ಯಂಕ, 32);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ibl = 0; ibl < nb; ++ibl) {
        ಸ್ಥಿರಾಂಕ int8_t  * q8 = y[ibl].qs;
        ಸ್ಥಿರಾಂಕ uint8_t * iq4 = x[ibl].qs;
        uint16_t h = x[ibl].scales_h;

        ಪೂರ್ಣಾಂಕ sumi = 0;

        #ಸೂಚನೆ GCC unroll 1
        // Process the entire super-block together.
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ib = 0; ib < QK_K / 256; ++ib) {
            // Weights and activations.
            ಸ್ಥಿರಾಂಕ vuint8m1_t iq4_packed = __riscv_vle8_v_u8m1(iq4, 128);
            iq4 += 128;

            // Unpack the weight blocks.
            ಸ್ಥಿರಾಂಕ vuint8m1_t iq4bits_lo = __riscv_vand_vx_u8m1(iq4_packed, 0xf, 128);
            ಸ್ಥಿರಾಂಕ vuint8m1_t iq4bits_hi = __riscv_vsrl_vx_u8m1(iq4_packed, 4, 128);
            ಸ್ಥಿರಾಂಕ vuint8m2_t iq4bits = __riscv_vcreate_v_u8m1_u8m2(iq4bits_lo, iq4bits_hi);
            ಸ್ಥಿರಾಂಕ vuint8m2_t iq4bits_reorder = __riscv_vreinterpret_v_u64m2_u8m2(__riscv_vrgatherei16_vv_u64m2(__riscv_vreinterpret_v_u8m2_u64m2(iq4bits), i_vec, 32));
            ಸ್ಥಿರಾಂಕ vint8m2_t iq4b = __riscv_vrgather_vv_i8m2(ಮೌಲ್ಯಗಳು, iq4bits_reorder, 256);

            __asm__ __volatile__("" ::: "memory");

            // Multiply with activations.
            ಸ್ಥಿರಾಂಕ vint8m2_t q8b = __riscv_vle8_v_i8m2(q8, 256);
            ಸ್ಥಿರಾಂಕ vint16m4_t prod = __riscv_vwmul_vv_i16m4(iq4b, q8b, 256);
            q8 += 256;

            // Mask for processing 32 elements per prod register.
            ಸ್ಥಿರಾಂಕ vuint16m1_t p_ಸೂಚ್ಯಂಕ = __riscv_vid_v_u16m1(64);
            ಸ್ಥಿರಾಂಕ vbool16_t p_ಮುಖವಾಡ = __riscv_vmsgtu_vx_u16m1_b16(p_ಸೂಚ್ಯಂಕ, 31, 64);

            // Reduce separately.
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ acc0 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(        __riscv_vget_v_i16m4_i16m1(prod, 0), __riscv_vmv_v_x_i32m1(0, 1), 32));
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ acc1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1_m(p_ಮುಖವಾಡ, __riscv_vget_v_i16m4_i16m1(prod, 0), __riscv_vmv_v_x_i32m1(0, 1), 64));
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ acc2 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(        __riscv_vget_v_i16m4_i16m1(prod, 1), __riscv_vmv_v_x_i32m1(0, 1), 32));
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ acc3 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1_m(p_ಮುಖವಾಡ, __riscv_vget_v_i16m4_i16m1(prod, 1), __riscv_vmv_v_x_i32m1(0, 1), 64));
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ acc4 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(        __riscv_vget_v_i16m4_i16m1(prod, 2), __riscv_vmv_v_x_i32m1(0, 1), 32));
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ acc5 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1_m(p_ಮುಖವಾಡ, __riscv_vget_v_i16m4_i16m1(prod, 2), __riscv_vmv_v_x_i32m1(0, 1), 64));
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ acc6 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(        __riscv_vget_v_i16m4_i16m1(prod, 3), __riscv_vmv_v_x_i32m1(0, 1), 32));
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ acc7 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1_m(p_ಮುಖವಾಡ, __riscv_vget_v_i16m4_i16m1(prod, 3), __riscv_vmv_v_x_i32m1(0, 1), 64));

            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ls0 = ((x[ibl].scales_l[0] & 0xf)  | ((h << 4) & 0x30)) - 32;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ls1 = ((x[ibl].scales_l[0] >>  4)  | ((h << 2) & 0x30)) - 32;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ls2 = ((x[ibl].scales_l[1] &  0xf) | ((h << 0) & 0x30)) - 32;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ls3 = ((x[ibl].scales_l[1] >>  4)  | ((h >> 2) & 0x30)) - 32;
            h >>= 8;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ls4 = ((x[ibl].scales_l[2] & 0xf)  | ((h << 4) & 0x30)) - 32;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ls5 = ((x[ibl].scales_l[2] >>  4)  | ((h << 2) & 0x30)) - 32;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ls6 = ((x[ibl].scales_l[3] &  0xf) | ((h << 0) & 0x30)) - 32;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ls7 = ((x[ibl].scales_l[3] >>  4)  | ((h >> 2) & 0x30)) - 32;

            sumi += acc0 * ls0;
            sumi += acc1 * ls1;
            sumi += acc2 * ls2;
            sumi += acc3 * ls3;
            sumi += acc4 * ls4;
            sumi += acc5 * ls5;
            sumi += acc6 * ls6;
            sumi += acc7 * ls7;

            __asm__ __volatile__("" ::: "memory");
        }

        sumf += GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[ibl].d) * y[ibl].d * (sumi);
    }

    *s = sumf;
}
#ಆದರೆ_ಅಂತ್ಯ

ಶೂನ್ಯ ggml_vec_dot_iq4_xs_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
#ಆದರೆ defined __riscv_v
    ಆಯ್ಕೆ (__riscv_vlenb() * 8) {
        ಸ್ಥಿತಿ 128:
            ggml_vec_dot_iq4_xs_q8_K_vl128(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 256:
            ggml_vec_dot_iq4_xs_q8_K_vl256(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 512:
            ggml_vec_dot_iq4_xs_q8_K_vl512(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 1024:
            ggml_vec_dot_iq4_xs_q8_K_vl1024(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ggml_vec_dot_iq4_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ggml_vec_dot_iq4_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

#ಆದರೆ defined __riscv_v
ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_tq1_0_q8_K_vl128(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_tq1_0 * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K  * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    ದಶಮಾಂಶ sumf = 0.0f;
    uint8_t ಘಾತ[16] = {1, 1, 1, 1, 3, 3, 3, 3, 9, 9, 9, 9, 27, 27, 27, 27};

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; i++) {
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT tq = x[i].qs;
        ಸ್ಥಿರಾಂಕ int8_t  * GGML_RESTRICT q8 = y[i].qs;

        // First loop.
        vint16m4_t suml1;
        {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vl = 32;
            ಸ್ಥಿರಾಂಕ vuint8m2_t tqb = __riscv_vle8_v_u8m2(tq, vl);
            tq += 32;

            {
                ಸ್ಥಿರಾಂಕ vuint16m4_t tq0 = __riscv_vsrl_vx_u16m4(__riscv_vwmulu_vx_u16m4(tqb, 3, vl), 8, vl);
                ಸ್ಥಿರಾಂಕ vint16m4_t q80 = __riscv_vwcvt_x_x_v_i16m4(__riscv_vle8_v_i8m2(q8, vl), vl);
                suml1 = __riscv_vmul_vv_i16m4(__riscv_vreinterpret_v_u16m4_i16m4(__riscv_vsub_vx_u16m4(tq0, 1, vl)), q80, vl);
                q8 += 32;
            }

            uint8_t pow3 = 3;
            #ಸೂಚನೆ GCC unroll 1
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < 4; t++) {
                ಸ್ಥಿರಾಂಕ vuint16m4_t tqn = __riscv_vsrl_vx_u16m4(__riscv_vwmulu_vx_u16m4(__riscv_vmul_vx_u8m2(tqb, pow3, vl), 3, vl), 8, vl);
                ಸ್ಥಿರಾಂಕ vint16m4_t q8n = __riscv_vwcvt_x_x_v_i16m4(__riscv_vle8_v_i8m2(q8, vl), vl);
                suml1 = __riscv_vmacc_vv_i16m4(suml1, __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vsub_vx_u16m4(tqn, 1, vl)), q8n, vl);
                pow3 *= 3;
                q8 += 32;
            }
        }

        // Second loop.
        vint16m2_t suml2;
        {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vl = 16;
            ಸ್ಥಿರಾಂಕ vuint8m1_t tqb = __riscv_vle8_v_u8m1(tq, vl);

            {
                ಸ್ಥಿರಾಂಕ vuint16m2_t tq0 = __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(tqb, 3, vl), 8, vl);
                ಸ್ಥಿರಾಂಕ vint16m2_t q80 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(q8, vl), vl);
                suml2 = __riscv_vmul_vv_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(tq0, 1, vl)), q80, vl);
                q8 += 16;
            }

            uint8_t pow3 = 3;
            #ಸೂಚನೆ GCC unroll 1
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < 4; t++) {
                ಸ್ಥಿರಾಂಕ vuint16m2_t tqn = __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(__riscv_vmul_vx_u8m1(tqb, pow3, vl), 3, vl), 8, vl);
                ಸ್ಥಿರಾಂಕ vint16m2_t q8n = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(q8, vl), vl);
                suml2 = __riscv_vmacc_vv_i16m2(suml2, __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(tqn, 1, vl)), q8n, vl);
                pow3 *= 3;
                q8 += 16;
            }
        }

        // Third loop.
        vint16m2_t suml3;
        {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vl = 16;

            uint32_t qh;
            ಸ್ಮರಣೆ_ನಕಲು(&qh, &x[i].qh[0], 4);
            // Prevent fusion with vmv.
            __asm__ __volatile__("" : "+r"(qh));
            ಸ್ಥಿರಾಂಕ vuint8m1_t tqb = __riscv_vreinterpret_v_u32m1_u8m1(__riscv_vmv_v_x_u32m1(qh, vl / 4));

            ಸ್ಥಿರಾಂಕ vuint8m1_t p = __riscv_vle8_v_u8m1(ಘಾತ, vl);

            ಸ್ಥಿರಾಂಕ vuint16m2_t tq0 = __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(__riscv_vmul_vv_u8m1(tqb, p, vl), 3, vl), 8, vl);

            ಸ್ಥಿರಾಂಕ vint16m2_t q80 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(q8, vl), vl);

            suml3 = __riscv_vmul_vv_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(tq0, 1, vl)), q80, vl);
        }

        vint16m2_t sumb = __riscv_vadd_vv_i16m2(__riscv_vget_v_i16m4_i16m2(suml1, 0), __riscv_vget_v_i16m4_i16m2(suml1, 1), 16);
        sumb = __riscv_vadd_vv_i16m2(sumb, suml2, 16);
        sumb = __riscv_vadd_vv_i16m2(sumb, suml3, 16);

        vint32m1_t ಮೊತ್ತ = __riscv_vwredsum_vs_i16m2_i32m1(sumb, __riscv_vmv_v_x_i32m1(0, 1), 16);
        sumf += __riscv_vmv_x_s_i32m1_i32(ಮೊತ್ತ) * y[i].d * GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d);
    }

    *s = sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_tq1_0_q8_K_vl256(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_tq1_0 * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K  * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    ದಶಮಾಂಶ sumf = 0.0f;
    uint8_t ಘಾತ[16] = {1, 1, 1, 1, 3, 3, 3, 3, 9, 9, 9, 9, 27, 27, 27, 27};

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; i++) {
        // First loop.
        vint16m2_t suml1;
        {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vl = 32;
            vuint8m1_t tq = __riscv_vle8_v_u8m1(x[i].qs, vl);

            vuint16m2_t tq0 = __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(tq, 3, vl), 8, vl);
            vuint16m2_t tq1 = __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(__riscv_vmul_vx_u8m1(tq, 3, vl), 3, vl), 8, vl);
            vuint16m2_t tq2 = __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(__riscv_vmul_vx_u8m1(tq, 9, vl), 3, vl), 8, vl);
            vuint16m2_t tq3 = __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(__riscv_vmul_vx_u8m1(tq, 27, vl), 3, vl), 8, vl);
            vuint16m2_t tq4 = __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(__riscv_vmul_vx_u8m1(tq, 81, vl), 3, vl), 8, vl);

            vint16m2_t q80 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(y[i].qs + 0, vl), vl);
            vint16m2_t q81 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(y[i].qs + 32, vl), vl);
            vint16m2_t q82 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(y[i].qs + 64, vl), vl);
            vint16m2_t q83 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(y[i].qs + 96, vl), vl);
            vint16m2_t q84 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(y[i].qs + 128, vl), vl);

            vint16m2_t sum0 = __riscv_vmul_vv_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(tq0, 1, vl)), q80, vl);
            vint16m2_t sum1 = __riscv_vmul_vv_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(tq1, 1, vl)), q81, vl);
            vint16m2_t sum2 = __riscv_vmul_vv_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(tq2, 1, vl)), q82, vl);
            vint16m2_t sum3 = __riscv_vmul_vv_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(tq3, 1, vl)), q83, vl);
            vint16m2_t sum4 = __riscv_vmul_vv_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(tq4, 1, vl)), q84, vl);

            vint16m2_t sumi0 = __riscv_vadd_vv_i16m2(sum0, sum1, vl);
            vint16m2_t sumi1 = __riscv_vadd_vv_i16m2(sum2, sum3, vl);
            suml1 = __riscv_vadd_vv_i16m2(sum4, __riscv_vadd_vv_i16m2(sumi0, sumi1, vl), vl);
        }

        // Second loop.
        vint16m1_t suml2;
        {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vl = 16;
            vuint8mf2_t tq = __riscv_vle8_v_u8mf2(x[i].qs + 32, vl);

            vuint16m1_t tq0 = __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(tq, 3 * 1, vl), 8, vl);
            vuint16m1_t tq1 = __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(__riscv_vmul_vx_u8mf2(tq, 3, vl), 3, vl), 8, vl);
            vuint16m1_t tq2 = __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(__riscv_vmul_vx_u8mf2(tq, 9, vl), 3, vl), 8, vl);
            vuint16m1_t tq3 = __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(__riscv_vmul_vx_u8mf2(tq, 27, vl), 3, vl), 8, vl);
            vuint16m1_t tq4 = __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(__riscv_vmul_vx_u8mf2(tq, 81, vl), 3, vl), 8, vl);

            vint16m1_t q80 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(y[i].qs + 160, vl), vl);
            vint16m1_t q81 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(y[i].qs + 176, vl), vl);
            vint16m1_t q82 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(y[i].qs + 192, vl), vl);
            vint16m1_t q83 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(y[i].qs + 208, vl), vl);
            vint16m1_t q84 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(y[i].qs + 224, vl), vl);

            vint16m1_t sum0 = __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(tq0, 1, vl)), q80, vl);
            vint16m1_t sum1 = __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(tq1, 1, vl)), q81, vl);
            vint16m1_t sum2 = __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(tq2, 1, vl)), q82, vl);
            vint16m1_t sum3 = __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(tq3, 1, vl)), q83, vl);
            vint16m1_t sum4 = __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(tq4, 1, vl)), q84, vl);

            vint16m1_t sumi0 = __riscv_vadd_vv_i16m1(sum0, sum1, vl);
            vint16m1_t sumi1 = __riscv_vadd_vv_i16m1(sum2, sum3, vl);
            suml2 = __riscv_vadd_vv_i16m1(sum4, __riscv_vadd_vv_i16m1(sumi0, sumi1, vl), vl);
        }

        // Third loop.
        vint16m1_t suml3;
        {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vl = 16;

            uint32_t qh;
            ಸ್ಮರಣೆ_ನಕಲು(&qh, &x[i].qh[0], 4);
            // Prevent fusion with vmv.
            __asm__ __volatile__("" : "+r"(qh));
            vuint8mf2_t tq = __riscv_vreinterpret_v_u32mf2_u8mf2(__riscv_vmv_v_x_u32mf2(qh, vl / 4));

            vuint8mf2_t p = __riscv_vle8_v_u8mf2(ಘಾತ, vl);

            vuint16m1_t tq0 = __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(__riscv_vmul_vv_u8mf2(tq, p, vl), 3, vl), 8, vl);

            vint16m1_t q80 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(y[i].qs + 240, vl), vl);

            suml3 = __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(tq0, 1, vl)), q80, vl);
        }

        vint16m1_t sumb = __riscv_vadd_vv_i16m1(__riscv_vget_v_i16m2_i16m1(suml1, 0), __riscv_vget_v_i16m2_i16m1(suml1, 1), 16);
        sumb = __riscv_vadd_vv_i16m1(sumb, __riscv_vadd_vv_i16m1(suml2, suml3, 16), 16);

        vint32m1_t ಮೊತ್ತ = __riscv_vwredsum_vs_i16m1_i32m1(sumb, __riscv_vmv_v_x_i32m1(0, 1), 16);
        sumf += __riscv_vmv_x_s_i32m1_i32(ಮೊತ್ತ) * y[i].d * GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d);
    }

    *s = sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_tq1_0_q8_K_vl512(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_tq1_0 * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K  * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    ದಶಮಾಂಶ sumf = 0.0f;
    uint8_t ಘಾತ[16] = {1, 1, 1, 1, 3, 3, 3, 3, 9, 9, 9, 9, 27, 27, 27, 27};

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; i++) {
        // First loop.
        vint16m1_t suml1;
        {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vl = 32;
            vuint8mf2_t tq = __riscv_vle8_v_u8mf2(x[i].qs, vl);

            vuint16m1_t tq0 = __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(tq, 3, vl), 8, vl);
            vuint16m1_t tq1 = __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(__riscv_vmul_vx_u8mf2(tq, 3, vl), 3, vl), 8, vl);
            vuint16m1_t tq2 = __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(__riscv_vmul_vx_u8mf2(tq, 9, vl), 3, vl), 8, vl);
            vuint16m1_t tq3 = __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(__riscv_vmul_vx_u8mf2(tq, 27, vl), 3, vl), 8, vl);
            vuint16m1_t tq4 = __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(__riscv_vmul_vx_u8mf2(tq, 81, vl), 3, vl), 8, vl);

            vint16m1_t q80 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(y[i].qs + 0, vl), vl);
            vint16m1_t q81 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(y[i].qs + 32, vl), vl);
            vint16m1_t q82 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(y[i].qs + 64, vl), vl);
            vint16m1_t q83 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(y[i].qs + 96, vl), vl);
            vint16m1_t q84 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(y[i].qs + 128, vl), vl);

            vint16m1_t sum0 = __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(tq0, 1, vl)), q80, vl);
            vint16m1_t sum1 = __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(tq1, 1, vl)), q81, vl);
            vint16m1_t sum2 = __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(tq2, 1, vl)), q82, vl);
            vint16m1_t sum3 = __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(tq3, 1, vl)), q83, vl);
            vint16m1_t sum4 = __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(tq4, 1, vl)), q84, vl);

            vint16m1_t sumi0 = __riscv_vadd_vv_i16m1(sum0, sum1, vl);
            vint16m1_t sumi1 = __riscv_vadd_vv_i16m1(sum2, sum3, vl);
            suml1 = __riscv_vadd_vv_i16m1(sum4, __riscv_vadd_vv_i16m1(sumi0, sumi1, vl), vl);
        }

        // Second loop.
        vint16mf2_t suml2;
        {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vl = 16;
            vuint8mf4_t tq = __riscv_vle8_v_u8mf4(x[i].qs + 32, vl);

            vuint16mf2_t tq0 = __riscv_vsrl_vx_u16mf2(__riscv_vwmulu_vx_u16mf2(tq, 3 * 1, vl), 8, vl);
            vuint16mf2_t tq1 = __riscv_vsrl_vx_u16mf2(__riscv_vwmulu_vx_u16mf2(__riscv_vmul_vx_u8mf4(tq, 3, vl), 3, vl), 8, vl);
            vuint16mf2_t tq2 = __riscv_vsrl_vx_u16mf2(__riscv_vwmulu_vx_u16mf2(__riscv_vmul_vx_u8mf4(tq, 9, vl), 3, vl), 8, vl);
            vuint16mf2_t tq3 = __riscv_vsrl_vx_u16mf2(__riscv_vwmulu_vx_u16mf2(__riscv_vmul_vx_u8mf4(tq, 27, vl), 3, vl), 8, vl);
            vuint16mf2_t tq4 = __riscv_vsrl_vx_u16mf2(__riscv_vwmulu_vx_u16mf2(__riscv_vmul_vx_u8mf4(tq, 81, vl), 3, vl), 8, vl);

            vint16mf2_t q80 = __riscv_vwcvt_x_x_v_i16mf2(__riscv_vle8_v_i8mf4(y[i].qs + 160, vl), vl);
            vint16mf2_t q81 = __riscv_vwcvt_x_x_v_i16mf2(__riscv_vle8_v_i8mf4(y[i].qs + 176, vl), vl);
            vint16mf2_t q82 = __riscv_vwcvt_x_x_v_i16mf2(__riscv_vle8_v_i8mf4(y[i].qs + 192, vl), vl);
            vint16mf2_t q83 = __riscv_vwcvt_x_x_v_i16mf2(__riscv_vle8_v_i8mf4(y[i].qs + 208, vl), vl);
            vint16mf2_t q84 = __riscv_vwcvt_x_x_v_i16mf2(__riscv_vle8_v_i8mf4(y[i].qs + 224, vl), vl);

            vint16mf2_t sum0 = __riscv_vmul_vv_i16mf2(__riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vsub_vx_u16mf2(tq0, 1, vl)), q80, vl);
            vint16mf2_t sum1 = __riscv_vmul_vv_i16mf2(__riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vsub_vx_u16mf2(tq1, 1, vl)), q81, vl);
            vint16mf2_t sum2 = __riscv_vmul_vv_i16mf2(__riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vsub_vx_u16mf2(tq2, 1, vl)), q82, vl);
            vint16mf2_t sum3 = __riscv_vmul_vv_i16mf2(__riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vsub_vx_u16mf2(tq3, 1, vl)), q83, vl);
            vint16mf2_t sum4 = __riscv_vmul_vv_i16mf2(__riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vsub_vx_u16mf2(tq4, 1, vl)), q84, vl);

            vint16mf2_t sumi0 = __riscv_vadd_vv_i16mf2(sum0, sum1, vl);
            vint16mf2_t sumi1 = __riscv_vadd_vv_i16mf2(sum2, sum3, vl);
            suml2 = __riscv_vadd_vv_i16mf2(sum4, __riscv_vadd_vv_i16mf2(sumi0, sumi1, vl), vl);
        }

        // Third loop.
        vint16mf2_t suml3;
        {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vl = 16;

            uint32_t qh;
            ಸ್ಮರಣೆ_ನಕಲು(&qh, &x[i].qh[0], 4);
            // Prevent fusion with vmv.
            __asm__ __volatile__("" : "+r"(qh));
            vuint8mf4_t tq = __riscv_vlmul_trunc_v_u8mf2_u8mf4(__riscv_vreinterpret_v_u32mf2_u8mf2(__riscv_vmv_v_x_u32mf2(qh, vl / 4)));

            vuint8mf4_t p = __riscv_vle8_v_u8mf4(ಘಾತ, vl);

            vuint16mf2_t tq0 = __riscv_vsrl_vx_u16mf2(__riscv_vwmulu_vx_u16mf2(__riscv_vmul_vv_u8mf4(tq, p, vl), 3, vl), 8, vl);

            vint16mf2_t q80 = __riscv_vwcvt_x_x_v_i16mf2(__riscv_vle8_v_i8mf4(y[i].qs + 240, vl), vl);

            suml3 = __riscv_vmul_vv_i16mf2(__riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vsub_vx_u16mf2(tq0, 1, vl)), q80, vl);
        }

        vint32m1_t ಮೊತ್ತ = __riscv_vwredsum_vs_i16m1_i32m1(suml1, __riscv_vmv_v_x_i32m1(0, 1), 32);
        ಮೊತ್ತ = __riscv_vwredsum_vs_i16mf2_i32m1(__riscv_vadd_vv_i16mf2(suml2, suml3, 16), ಮೊತ್ತ, 16);
        sumf += __riscv_vmv_x_s_i32m1_i32(ಮೊತ್ತ) * y[i].d * GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d);
    }

    *s = sumf;
}
#ಆದರೆ_ಅಂತ್ಯ

ಶೂನ್ಯ ggml_vec_dot_tq1_0_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
#ಆದರೆ defined __riscv_v
    ಆಯ್ಕೆ (__riscv_vlenb() * 8) {
        ಸ್ಥಿತಿ 128:
            ggml_vec_dot_tq1_0_q8_K_vl128(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 256:
            ggml_vec_dot_tq1_0_q8_K_vl256(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: // 512 and above
            ggml_vec_dot_tq1_0_q8_K_vl512(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ggml_vec_dot_tq1_0_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

#ಆದರೆ defined __riscv_v
ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_tq2_0_q8_K_vl128(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_tq2_0 * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K  * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;
    ದಶಮಾಂಶ sumf = 0.0f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        int32_t sumi = 0;

        ಗಾಗಿ (size_t j = 0; j < ಗಾತ್ರ(x[0].qs); j += 32) {
            ಸ್ಥಿರಾಂಕ int8_t * py0 = &y[i].qs[j * 4 + 0 * 32];
            ಸ್ಥಿರಾಂಕ int8_t * py1 = &y[i].qs[j * 4 + 1 * 32];
            ಸ್ಥಿರಾಂಕ int8_t * py2 = &y[i].qs[j * 4 + 2 * 32];
            ಸ್ಥಿರಾಂಕ int8_t * py3 = &y[i].qs[j * 4 + 3 * 32];
            ಸ್ಥಿರಾಂಕ uint8_t* px  = &x[i].qs[j];

            size_t vl = __riscv_vsetvl_e16m4(32);
            vint16m4_t vacc16 = __riscv_vmv_v_x_i16m4(0, vl);

            // Load Raw Packed elements
            vl = __riscv_vsetvl_e8m2(32);
            vuint8m2_t vx_u8 = __riscv_vle8_v_u8m2(px, vl);

            // Process bits 1:0
            {
                // Unpack
                vuint8m2_t t0 = __riscv_vand_vx_u8m2(vx_u8, 0x03, vl);
                vint8m2_t vq = __riscv_vsub_vx_i8m2(__riscv_vreinterpret_v_u8m2_i8m2(t0), 1, vl);
                vint8m2_t vy = __riscv_vle8_v_i8m2(py0, vl);
                // Accumulate
                vacc16 = __riscv_vwmacc_vv_i16m4(vacc16, vq, vy, vl);
            }
            __asm__ ಅಸ್ಥಿರ("" ::: "memory");
            // Process bits 3:2
            {
                vuint8m2_t t1 = __riscv_vsrl_vx_u8m2(vx_u8, 2, vl);
                t1 = __riscv_vand_vx_u8m2(t1, 0x03, vl);
                vint8m2_t vq = __riscv_vsub_vx_i8m2(__riscv_vreinterpret_v_u8m2_i8m2(t1), 1, vl);

                vint8m2_t vy = __riscv_vle8_v_i8m2(py1, vl);
                vacc16 = __riscv_vwmacc_vv_i16m4(vacc16, vq, vy, vl);
            }
            __asm__ ಅಸ್ಥಿರ("" ::: "memory");
            // Process bits 5:4
            {
                vuint8m2_t t2 = __riscv_vsrl_vx_u8m2(vx_u8, 4, vl);
                t2 = __riscv_vand_vx_u8m2(t2, 0x03, vl);
                vint8m2_t vq = __riscv_vsub_vx_i8m2(__riscv_vreinterpret_v_u8m2_i8m2(t2), 1, vl);

                vint8m2_t vy = __riscv_vle8_v_i8m2(py2, vl);
                vacc16 = __riscv_vwmacc_vv_i16m4(vacc16, vq, vy, vl);
            }
            __asm__ ಅಸ್ಥಿರ("" ::: "memory");
            // Process bits 7:6
            {
                vuint8m2_t t3 = __riscv_vsrl_vx_u8m2(vx_u8, 6, vl);
                vint8m2_t vq = __riscv_vsub_vx_i8m2(__riscv_vreinterpret_v_u8m2_i8m2(t3), 1, vl);

                vint8m2_t vy = __riscv_vle8_v_i8m2(py3, vl);
                vacc16 = __riscv_vwmacc_vv_i16m4(vacc16, vq, vy, vl);
            }
            __asm__ ಅಸ್ಥಿರ("" ::: "memory");
            vl = __riscv_vsetvl_e16m4(32);
            vint32m1_t vzero32 = __riscv_vmv_v_x_i32m1(0, 1);
            vint32m1_t vred32 = __riscv_vwredsum_vs_i16m4_i32m1(vacc16, vzero32, vl);
            sumi += __riscv_vmv_x_s_i32m1_i32(vred32);
        }

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = y[i].d * GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d);
        sumf += (ದಶಮಾಂಶ)sumi * d;
    }

    *s = sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_tq2_0_q8_K_vl256(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_tq2_0 * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K  * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    ದಶಮಾಂಶ sumf = 0.0f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        int32_t sumi = 0;

        ಗಾಗಿ (size_t j = 0; j < ಗಾತ್ರ(x[0].qs); j += 32) {
            ಸ್ಥಿರಾಂಕ int8_t * py0 = &y[i].qs[j * 4 + 0 * 32];
            ಸ್ಥಿರಾಂಕ int8_t * py1 = &y[i].qs[j * 4 + 1 * 32];
            ಸ್ಥಿರಾಂಕ int8_t * py2 = &y[i].qs[j * 4 + 2 * 32];
            ಸ್ಥಿರಾಂಕ int8_t * py3 = &y[i].qs[j * 4 + 3 * 32];
            ಸ್ಥಿರಾಂಕ uint8_t* px  = &x[i].qs[j];

            size_t vlmax_16m2 = __riscv_vsetvl_e16m2(32);
            vint16m2_t vacc16 = __riscv_vmv_v_x_i16m2(0, vlmax_16m2);

            size_t vl = __riscv_vsetvl_e8m1(32);

            vuint8m1_t vx_u8 = __riscv_vle8_v_u8m1(px, vl);

            vint8m1_t vy0 = __riscv_vle8_v_i8m1(py0 , vl);
            vint8m1_t vy1 = __riscv_vle8_v_i8m1(py1, vl);
            vint8m1_t vy2 = __riscv_vle8_v_i8m1(py2, vl);
            vint8m1_t vy3 = __riscv_vle8_v_i8m1(py3, vl);

            // l=0 (bits 1:0)
            vuint8m1_t t0 = __riscv_vand_vx_u8m1(vx_u8, 0x03, vl);
            vint8m1_t vq0 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(t0), 1, vl);

            // l=1 (bits 3:2)
            vuint8m1_t t1 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vx_u8, 2, vl), 0x03, vl);
            vint8m1_t vq1 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(t1), 1, vl);

            // l=2 (bits 5:4)
            vuint8m1_t t2 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vx_u8, 4, vl), 0x03, vl);
            vint8m1_t vq2 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(t2), 1, vl);

            // l=3 (bits 7:6)
            vuint8m1_t t3 = __riscv_vsrl_vx_u8m1(vx_u8, 6, vl); // No final AND needed as vsrl shifts in zeros
            vint8m1_t vq3 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(t3), 1, vl);

            // 4. Multiply and accumulate
            vacc16 = __riscv_vwmacc_vv_i16m2(vacc16, vq0, vy0, vl);
            vacc16 = __riscv_vwmacc_vv_i16m2(vacc16, vq1, vy1, vl);
            vacc16 = __riscv_vwmacc_vv_i16m2(vacc16, vq2, vy2, vl);
            vacc16 = __riscv_vwmacc_vv_i16m2(vacc16, vq3, vy3, vl);

            vlmax_16m2 = __riscv_vsetvl_e16m2(32);
            vint32m1_t vzero32 = __riscv_vmv_v_x_i32m1(0, 1);
            vint32m1_t vred32 = __riscv_vwredsum_vs_i16m2_i32m1(vacc16, vzero32, vlmax_16m2);

            sumi += __riscv_vmv_x_s_i32m1_i32(vred32);
        }
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = y[i].d * GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d);
        sumf += (ದಶಮಾಂಶ)sumi * d;
    }

    *s = sumf;
}
#ಆದರೆ_ಅಂತ್ಯ

ಶೂನ್ಯ ggml_vec_dot_tq2_0_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
#ಆದರೆ defined __riscv_v
    ಆಯ್ಕೆ (__riscv_vlenb() * 8) {
        ಸ್ಥಿತಿ 128:
            ggml_vec_dot_tq2_0_q8_K_vl128(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: // 256 and above
            ggml_vec_dot_tq2_0_q8_K_vl256(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ggml_vec_dot_tq2_0_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

#ಆದರೆ defined __riscv_v
ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_mxfp4_q8_0_vl128(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);
    ಪ್ರತಿಜ್ಞಾ(n % QK_MXFP4 == 0);
    static_assert(QK_MXFP4 == QK8_0, "QK_MXFP4 and QK8_0 must be the same");

    ಸ್ಥಿರಾಂಕ ಖಂಡ_mxfp4 * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0  * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_MXFP4;

    ಪೂರ್ಣಾಂಕ ib = 0;
    ದಶಮಾಂಶ sumf = 0;

    // Load the lookup table once.
    ಸ್ಥಿರಾಂಕ vint8m2_t ಮೌಲ್ಯಗಳು = __riscv_vle8_v_i8m2(kvalues_mxfp4, 16);
    ಪೂರ್ಣಾಂಕ acc1, acc2;

    // We process 2 blocks at once.
    ಗಾಗಿ (; ib + 1 < nb; ib += 2) {
        // Weights and activations.
        vuint8m1_t mx_packed1 = __riscv_vle8_v_u8m1(x[ib + 0].qs, 16);
        vint8m2_t q8b1 = __riscv_vle8_v_i8m2(y[ib + 0].qs, 32);
        vuint8m1_t mx_packed2 = __riscv_vle8_v_u8m1(x[ib + 1].qs, 16);
        vint8m2_t q8b2 = __riscv_vle8_v_i8m2(y[ib + 1].qs, 32);

        // Unpack the weight blocks.
        vuint8m2_t mxbits1 = __riscv_vcreate_v_u8m1_u8m2(
            __riscv_vand_vx_u8m1(mx_packed1, 0xf, 16),
            __riscv_vsrl_vx_u8m1(mx_packed1, 4, 16)
        );
        vuint8m2_t mxbits2 = __riscv_vcreate_v_u8m1_u8m2(
            __riscv_vand_vx_u8m1(mx_packed2, 0xf, 16),
            __riscv_vsrl_vx_u8m1(mx_packed2, 4, 16)
        );

        // Gather values from the lookup table.
        vint8m2_t mxb1 = __riscv_vrgather_vv_i8m2(ಮೌಲ್ಯಗಳು, mxbits1, 32);
        vint8m2_t mxb2 = __riscv_vrgather_vv_i8m2(ಮೌಲ್ಯಗಳು, mxbits2, 32);

        // Accumulation.
        vint16m4_t sum1 = __riscv_vwmul_vv_i16m4(q8b1, mxb1, 32);
        vint16m4_t sum2 = __riscv_vwmul_vv_i16m4(q8b2, mxb2, 32);
        __riscv_vse32_v_i32m1(&acc1,__riscv_vwredsum_vs_i16m4_i32m1(sum1, __riscv_vmv_v_x_i32m1(0, 1), 32), 1);
        __riscv_vse32_v_i32m1(&acc2,__riscv_vwredsum_vs_i16m4_i32m1(sum2, __riscv_vmv_v_x_i32m1(0, 1), 32), 1);
        sumf += ((GGML_E8_M0_ಗಮ್ಯ_FP32_HALF(x[ib + 0].e) * GGML_CPU_FP16_ಗಮ್ಯ_FP32(y[ib + 0].d) * acc1));
        sumf += ((GGML_E8_M0_ಗಮ್ಯ_FP32_HALF(x[ib + 1].e) * GGML_CPU_FP16_ಗಮ್ಯ_FP32(y[ib + 1].d) * acc2));
    }

    *s = sumf;
}

ಸ್ಥಿರ NOINLINE ಶೂನ್ಯ ggml_vec_dot_mxfp4_q8_0_vl256(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);
    ಪ್ರತಿಜ್ಞಾ(n % QK_MXFP4 == 0);
    static_assert(QK_MXFP4 == QK8_0, "QK_MXFP4 and QK8_0 must be the same");

    ಸ್ಥಿರಾಂಕ ಖಂಡ_mxfp4 * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0  * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_MXFP4;

    ಪೂರ್ಣಾಂಕ ib = 0;
    ದಶಮಾಂಶ sumf = 0;

    // Load the lookup table once.
    ಸ್ಥಿರಾಂಕ vint8mf2_t ಮೌಲ್ಯಗಳು = __riscv_vle8_v_i8mf2(kvalues_mxfp4, 16);
    ಪೂರ್ಣಾಂಕ acc1, acc2;

    // We process 2 blocks at once.
    ಗಾಗಿ (; ib + 1 < nb; ib+=2) {
        // Weights and activations.
        vuint8mf2_t mx_packed1 = __riscv_vle8_v_u8mf2(x[ib + 0].qs, 16);
        vint8mf2_t q8b_lo1 = __riscv_vle8_v_i8mf2(y[ib + 0].qs, 16);
        vint8mf2_t q8b_hi1 = __riscv_vle8_v_i8mf2(y[ib + 0].qs + 16, 16);
        vuint8mf2_t mx_packed2 = __riscv_vle8_v_u8mf2(x[ib + 1].qs, 16);
        vint8mf2_t q8b_lo2 = __riscv_vle8_v_i8mf2(y[ib + 1].qs, 16);
        vint8mf2_t q8b_hi2 = __riscv_vle8_v_i8mf2(y[ib + 1].qs + 16, 16);

        // Unpack the weight blocks.
        vuint8mf2_t mxbits_lo1 = __riscv_vand_vx_u8mf2(mx_packed1, 0xf, 16);
        vuint8mf2_t mxbits_hi1 = __riscv_vsrl_vx_u8mf2(mx_packed1, 4, 16);
        vuint8mf2_t mxbits_lo2 = __riscv_vand_vx_u8mf2(mx_packed2, 0xf, 16);
        vuint8mf2_t mxbits_hi2 = __riscv_vsrl_vx_u8mf2(mx_packed2, 4, 16);

        // Gather values from the lookup table.
        vint8mf2_t mxb_lo1 = __riscv_vrgather_vv_i8mf2(ಮೌಲ್ಯಗಳು, mxbits_lo1, 16);
        vint8mf2_t mxb_hi1 = __riscv_vrgather_vv_i8mf2(ಮೌಲ್ಯಗಳು, mxbits_hi1, 16);
        vint8mf2_t mxb_lo2 = __riscv_vrgather_vv_i8mf2(ಮೌಲ್ಯಗಳು, mxbits_lo2, 16);
        vint8mf2_t mxb_hi2 = __riscv_vrgather_vv_i8mf2(ಮೌಲ್ಯಗಳು, mxbits_hi2, 16);

        // Accumulation.
        vint16m1_t sum1 = __riscv_vwmul_vv_i16m1(q8b_lo1, mxb_lo1, 16);
        sum1 = __riscv_vwmacc_vv_i16m1(sum1, q8b_hi1, mxb_hi1, 16);
        vint16m1_t sum2 = __riscv_vwmul_vv_i16m1(q8b_lo2, mxb_lo2, 16);
        sum2 = __riscv_vwmacc_vv_i16m1(sum2, q8b_hi2, mxb_hi2, 16);
        __riscv_vse32_v_i32m1(&acc1,__riscv_vwredsum_vs_i16m1_i32m1(sum1, __riscv_vmv_v_x_i32m1(0, 1), 16), 1);
        __riscv_vse32_v_i32m1(&acc2,__riscv_vwredsum_vs_i16m1_i32m1(sum2, __riscv_vmv_v_x_i32m1(0, 1), 16), 1);
        sumf += ((GGML_E8_M0_ಗಮ್ಯ_FP32_HALF(x[ib + 0].e) * GGML_CPU_FP16_ಗಮ್ಯ_FP32(y[ib + 0].d) * acc1));
        sumf += ((GGML_E8_M0_ಗಮ್ಯ_FP32_HALF(x[ib + 1].e) * GGML_CPU_FP16_ಗಮ್ಯ_FP32(y[ib + 1].d) * acc2));
    }

    *s = sumf;
}
#ಆದರೆ_ಅಂತ್ಯ

ಶೂನ್ಯ ggml_vec_dot_mxfp4_q8_0(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
#ಆದರೆ defined __riscv_v
    ಆಯ್ಕೆ (__riscv_vlenb() * 8) {
        ಸ್ಥಿತಿ 128:
            ggml_vec_dot_mxfp4_q8_0_vl128(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: // 256 and above
            ggml_vec_dot_mxfp4_q8_0_vl256(n, s, bs, vx, bx, vy, by, nrc);
            ನಿಲ್ಲಿಸು;
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ggml_vec_dot_mxfp4_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}
