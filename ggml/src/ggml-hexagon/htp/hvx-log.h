#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HVX_LOG_H
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_LOG_H

#ಅನ್ತರ್ಭಾವ "hvx-base.h"

// Approximates ln(x) element-wise for float vectors.
// x must contain positive float elements.
// Uses Abramowitz & Stegun polynomial approximation 4.1.44 for ln(1+y) over [0, 1].
ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_log_f32(HVX_ಸದಿಶ x) {
    // x = m * 2^e, where m in [1, 2)
    HVX_ಸದಿಶ biased_e = Q6_Vuw_vlsr_VuwR(x, 23);
    HVX_ಸದಿಶ e_int = Q6_Vw_vsub_VwVw(biased_e, Q6_V_vsplat_R(127));
    HVX_ಸದಿಶ e_float = Q6_Vsf_equals_Vw(e_int);

    // Extract mantissa and set exponent to 127 (which represents float value in [1.0, 2.0))
    HVX_ಸದಿಶ mant_ಮುಖವಾಡ = Q6_V_vsplat_R(0x007FFFFF);
    HVX_ಸದಿಶ exp_127 = Q6_V_vsplat_R(0x3F800000);
    HVX_ಸದಿಶ m = Q6_V_vor_VV(Q6_V_vand_VV(x, mant_ಮುಖವಾಡ), exp_127);

    // y = m - 1.0f, y in [0, 1)
    HVX_ಸದಿಶ y = hvx_vec_sub_f32_f32(m, hvx_vec_splat_f32(1.0f));

    // Abramowitz & Stegun 4.1.44 polynomial approximation of ln(1+y)
    HVX_ಸದಿಶ c;
    HVX_ಸದಿಶ res;

    c   = hvx_vec_splat_f32(-0.0064535442f);
    res = hvx_vec_mul_f32_f32(y, c);

    c   = hvx_vec_splat_f32(0.0360884937f);
    res = hvx_vec_ಕೂಡಿಸು_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(-0.0953293897f);
    res = hvx_vec_ಕೂಡಿಸು_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(0.1676540711f);
    res = hvx_vec_ಕೂಡಿಸು_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(-0.2407338084f);
    res = hvx_vec_ಕೂಡಿಸು_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(0.3317990258f);
    res = hvx_vec_ಕೂಡಿಸು_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(-0.4998741238f);
    res = hvx_vec_ಕೂಡಿಸು_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(0.9999964239f);
    res = hvx_vec_ಕೂಡಿಸು_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    // ln(x) = e * ln(2) + ln(1+y)
    HVX_ಸದಿಶ ln2 = hvx_vec_splat_f32(0.69314718056f);
    HVX_ಸದಿಶ ಪದ_e = hvx_vec_mul_f32_f32(e_float, ln2);

    ಹಿಂತಿರುಗಿ hvx_vec_ಕೂಡಿಸು_f32_f32(ಪದ_e, res);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_log_f32_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);

    HVX_ಸದಿಶ * ಸೀಮಿತ vdst = (HVX_ಸದಿಶ *) dst;
    HVX_ಸದಿಶ * ಸೀಮಿತ vsrc = (HVX_ಸದಿಶ *) src;

    ಸ್ಥಿರಾಂಕ uint32_t elem_ಗಾತ್ರ = ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t epv       = 128 / elem_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t nvec      = n / epv;
    ಸ್ಥಿರಾಂಕ uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ಗಾಗಿ (; i < nvec; i++) {
        vdst[i] = hvx_vec_log_f32(vsrc[i]);
    }
    ಆದರೆ (nloe) {
        HVX_ಸದಿಶ v = hvx_vec_log_f32(vsrc[i]);
        hvx_vec_store_a((ಶೂನ್ಯ *) &vdst[i], nloe * elem_ಗಾತ್ರ, v);
    }
}

// Compute log(x) for f16 by promoting to f32, applying hvx_vec_log_f32, and narrowing back.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_log_f16_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);

    HVX_ಸದಿಶ * ಸೀಮಿತ vdst = (HVX_ಸದಿಶ *) dst;
    HVX_ಸದಿಶ * ಸೀಮಿತ vsrc = (HVX_ಸದಿಶ *) src;

    ಸ್ಥಿರಾಂಕ uint32_t nvec = n / VLEN_FP16;
    ಸ್ಥಿರಾಂಕ uint32_t nloe = n % VLEN_FP16;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ಗಾಗಿ (; i < nvec; i++) {
        HVX_ಸದಿಶ_Pair p = hvx_vec_f16_ಗಮ್ಯ_f32(vsrc[i]);
        HVX_ಸದಿಶ r0 = hvx_vec_log_f32(Q6_V_lo_W(p));
        HVX_ಸದಿಶ r1 = hvx_vec_log_f32(Q6_V_hi_W(p));
        vdst[i] = hvx_vec_f32_ಗಮ್ಯ_f16(r0, r1);
    }
    ಆದರೆ (nloe) {
        HVX_ಸದಿಶ_Pair p = hvx_vec_f16_ಗಮ್ಯ_f32(vsrc[i]);
        HVX_ಸದಿಶ r0 = hvx_vec_log_f32(Q6_V_lo_W(p));
        HVX_ಸದಿಶ r1 = hvx_vec_log_f32(Q6_V_hi_W(p));
        HVX_ಸದಿಶ v = hvx_vec_f32_ಗಮ್ಯ_f16(r0, r1);
        hvx_vec_store_a((ಶೂನ್ಯ *) &vdst[i], nloe * SIZEOF_FP16, v);
    }
}

#ಆದರೆ_ಅಂತ್ಯ /* HVX_LOG_H */
