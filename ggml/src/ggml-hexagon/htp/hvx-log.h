#अगर_अघोषित HVX_LOG_H
#घोषणा HVX_LOG_H

#समावेश "hvx-base.h"

// Approximates ln(x) element-wise for float vectors.
// x must contain positive float elements.
// Uses Abramowitz & Stegun polynomial approximation 4.1.44 for ln(1+y) over [0, 1].
अटल अंतरभूत HVX_Vector hvx_vec_log_f32(HVX_Vector x) {
    // x = m * 2^e, where m in [1, 2)
    HVX_Vector biased_e = Q6_Vuw_vlsr_VuwR(x, 23);
    HVX_Vector e_int = Q6_Vw_vsub_VwVw(biased_e, Q6_V_vsplat_R(127));
    HVX_Vector e_float = Q6_Vsf_equals_Vw(e_int);

    // Extract mantissa and set exponent to 127 (which represents float value in [1.0, 2.0))
    HVX_Vector mant_mask = Q6_V_vsplat_R(0x007FFFFF);
    HVX_Vector exp_127 = Q6_V_vsplat_R(0x3F800000);
    HVX_Vector m = Q6_V_vor_VV(Q6_V_vand_VV(x, mant_mask), exp_127);

    // y = m - 1.0f, y in [0, 1)
    HVX_Vector y = hvx_vec_sub_f32_f32(m, hvx_vec_splat_f32(1.0f));

    // Abramowitz & Stegun 4.1.44 polynomial approximation of ln(1+y)
    HVX_Vector c;
    HVX_Vector res;

    c   = hvx_vec_splat_f32(-0.0064535442f);
    res = hvx_vec_mul_f32_f32(y, c);

    c   = hvx_vec_splat_f32(0.0360884937f);
    res = hvx_vec_add_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(-0.0953293897f);
    res = hvx_vec_add_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(0.1676540711f);
    res = hvx_vec_add_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(-0.2407338084f);
    res = hvx_vec_add_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(0.3317990258f);
    res = hvx_vec_add_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(-0.4998741238f);
    res = hvx_vec_add_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(0.9999964239f);
    res = hvx_vec_add_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    // ln(x) = e * ln(2) + ln(1+y)
    HVX_Vector ln2 = hvx_vec_splat_f32(0.69314718056f);
    HVX_Vector term_e = hvx_vec_mul_f32_f32(e_float, ln2);

    वापस hvx_vec_add_f32_f32(term_e, res);
}

अटल अंतरभूत व्योम hvx_log_f32_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    स्थिर uint32_t elem_आकार = माप(भग्न);
    स्थिर uint32_t epv       = 128 / elem_आकार;
    स्थिर uint32_t nvec      = n / epv;
    स्थिर uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    क्रम (; i < nvec; i++) {
        vdst[i] = hvx_vec_log_f32(vsrc[i]);
    }
    अगर (nloe) {
        HVX_Vector v = hvx_vec_log_f32(vsrc[i]);
        hvx_vec_store_a((व्योम *) &vdst[i], nloe * elem_आकार, v);
    }
}

// Compute log(x) for f16 by promoting to f32, applying hvx_vec_log_f32, and narrowing back.
अटल अंतरभूत व्योम hvx_log_f16_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    स्थिर uint32_t nvec = n / VLEN_FP16;
    स्थिर uint32_t nloe = n % VLEN_FP16;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    क्रम (; i < nvec; i++) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(vsrc[i]);
        HVX_Vector r0 = hvx_vec_log_f32(Q6_V_lo_W(p));
        HVX_Vector r1 = hvx_vec_log_f32(Q6_V_hi_W(p));
        vdst[i] = hvx_vec_f32_to_f16(r0, r1);
    }
    अगर (nloe) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(vsrc[i]);
        HVX_Vector r0 = hvx_vec_log_f32(Q6_V_lo_W(p));
        HVX_Vector r1 = hvx_vec_log_f32(Q6_V_hi_W(p));
        HVX_Vector v = hvx_vec_f32_to_f16(r0, r1);
        hvx_vec_store_a((व्योम *) &vdst[i], nloe * SIZEOF_FP16, v);
    }
}

#पूर्ण_अगर /* HVX_LOG_H */
