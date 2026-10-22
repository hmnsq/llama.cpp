#ifndef HVX_ERF_H
#define HVX_ERF_H

#അന്തര്ഭാവ "hvx-base.h"
#അന്തര്ഭാവ "hvx-exp.h"
#അന്തര്ഭാവ "hvx-inverse.h"

// Maximum error is about 1.5e-7 for the Abramowitz-Stegun approximation.
സ്ഥിതം __attribute__((noinline)) HVX_Vector hvx_vec_erf_f32(HVX_Vector x) {
    സ്ഥിരം HVX_Vector zero = hvx_vec_splat_f32(0.0f);
    സ്ഥിരം HVX_Vector ax = hvx_vec_abs_f32(x);
    HVX_Vector t = hvx_vec_inverse_f32(hvx_vec_add_f32_f32(
        hvx_vec_splat_f32(1.0f), hvx_vec_mul_f32_f32(hvx_vec_splat_f32(0.3275911f), ax)));

    HVX_Vector poly = hvx_vec_splat_f32(1.061405429f);
    poly = hvx_vec_add_f32_f32(hvx_vec_splat_f32(-1.453152027f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_add_f32_f32(hvx_vec_splat_f32(1.421413741f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_add_f32_f32(hvx_vec_splat_f32(-0.284496736f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_add_f32_f32(hvx_vec_splat_f32(0.254829592f), hvx_vec_mul_f32_f32(poly, t));

    സ്ഥിരം HVX_Vector exp_term = hvx_vec_exp_f32(hvx_vec_neg_f32(hvx_vec_mul_f32_f32(ax, ax)));
    HVX_Vector result = hvx_vec_sub_f32_f32(hvx_vec_splat_f32(1.0f),
        hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(poly, t), exp_term));

    സ്ഥിരം HVX_VectorPred neg = Q6_Q_vcmp_gt_VsfVsf(zero, x);
    result = Q6_V_vmux_QVV(neg, hvx_vec_neg_f32(result), result);
    തിരികെ result;
}

// GELU_ERF uses the erf definition.
സ്ഥിതം വരിയിൽ HVX_Vector hvx_vec_gelu_erf_f32(HVX_Vector x) {
    സ്ഥിരം HVX_Vector scale = hvx_vec_splat_f32(0.7071067811865475f);
    സ്ഥിരം HVX_Vector half  = hvx_vec_splat_f32(0.5f);
    സ്ഥിരം HVX_Vector one   = hvx_vec_splat_f32(1.0f);
    സ്ഥിരം HVX_Vector max_x = hvx_vec_splat_f32(10.0f);
    സ്ഥിരം HVX_Vector min_x = hvx_vec_splat_f32(-10.0f);
    സ്ഥിരം HVX_VectorPred neg_large = Q6_Q_vcmp_gt_VsfVsf(min_x, x);
    സ്ഥിരം HVX_VectorPred pos_large = Q6_Q_vcmp_gt_VsfVsf(x, max_x);
    HVX_Vector x_calc = Q6_V_vmux_QVV(neg_large, min_x, x);
    x_calc = Q6_V_vmux_QVV(pos_large, max_x, x_calc);
    സ്ഥിരം HVX_Vector erf = hvx_vec_erf_f32(hvx_vec_mul_f32_f32(x_calc, scale));

    HVX_Vector result = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(half, x_calc), hvx_vec_add_f32_f32(one, erf));

    result = Q6_V_vmux_QVV(neg_large, hvx_vec_splat_f32(0.0f), result);
    result = Q6_V_vmux_QVV(pos_large, x, result);
    തിരികെ result;
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_gelu_erf_f32_aa(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    സ്ഥിരം uint32_t elem_size = വലുപ്പം(ദശാംശം);
    സ്ഥിരം uint32_t epv       = 128 / elem_size;
    സ്ഥിരം uint32_t nvec      = n / epv;
    സ്ഥിരം uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    വേണ്ടി (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_erf_f32(vsrc[i]);
    }
    എങ്കിൽ (nloe) {
        HVX_Vector v = hvx_vec_gelu_erf_f32(vsrc[i]);
        hvx_vec_store_a((ശൂന്യം *) &vdst[i], nloe * elem_size, v);
    }
}

#endif /* HVX_ERF_H */
