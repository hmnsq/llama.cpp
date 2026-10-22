#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HVX_SIN_COS_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_SIN_COS_H

#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-base.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-floor.h"

// Range-reduce x to y in [-pi/2, pi/2] and the quadrant sign (-1)^n.
// Floor/truncate need IEEE bits, so convert qf32 back to sf before them.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_vec_sincos_reduce_f32(HVX_ਸਦਿਸ਼ x, HVX_ਸਦਿਸ਼ * y, HVX_ਸਦਿਸ਼ * sign) {
    HVX_ਸਦਿਸ਼ const_inv_pi  = hvx_vec_splat_f32(0.3183098861837907f);
    HVX_ਸਦਿਸ਼ const_half    = hvx_vec_splat_f32(0.5f);
    HVX_ਸਦਿਸ਼ const_pi      = hvx_vec_splat_f32(3.141592653589793f);
    HVX_ਸਦਿਸ਼ const_one     = hvx_vec_splat_f32(1.0f);
    HVX_ਸਦਿਸ਼ const_neg_one = hvx_vec_splat_f32(-1.0f);
    HVX_ਸਦਿਸ਼ const_one_i   = Q6_V_vsplat_R(1);

    HVX_ਸਦਿਸ਼ x_over_pi = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x, const_inv_pi));
    x_over_pi = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(x_over_pi, const_half));

    HVX_ਸਦਿਸ਼ n_float = hvx_vec_floor_f32(x_over_pi);
    HVX_ਸਦਿਸ਼ n_int   = hvx_vec_truncate_f32(n_float);

    HVX_ਸਦਿਸ਼ n_pi = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(n_float, const_pi));
    *y = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(x, n_pi));

    HVX_ਸਦਿਸ਼_Pred is_odd = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(n_int, const_one_i), const_one_i);
    *sign = Q6_V_vmux_QVV(is_odd, const_neg_one, const_one);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_vec_sincos_f32(HVX_ਸਦਿਸ਼ x, HVX_ਸਦਿਸ਼ * vcos, HVX_ਸਦਿਸ਼ * vsin) {
    HVX_ਸਦਿਸ਼ y;
    HVX_ਸਦਿਸ਼ sign;
    hvx_vec_sincos_reduce_f32(x, &y, &sign);

    HVX_ਸਦਿਸ਼ z = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(y, y));

    HVX_ਸਦਿਸ਼ c4 = hvx_vec_splat_f32(2.3557242013849433e-05f);
    HVX_ਸਦਿਸ਼ c3 = hvx_vec_splat_f32(-0.0013871428263450528f);
    HVX_ਸਦਿਸ਼ c2 = hvx_vec_splat_f32(0.041665895266688284f);
    HVX_ਸਦਿਸ਼ c1 = hvx_vec_splat_f32(-0.4999999360426369f);
    HVX_ਸਦਿਸ਼ c0 = hvx_vec_splat_f32(0.9999999999071725f);

    HVX_ਸਦਿਸ਼ cos_y = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(c3, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(z, c4))));
    cos_y = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(c2, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(z, cos_y))));
    cos_y = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(c1, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(z, cos_y))));
    cos_y = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(c0, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(z, cos_y))));

    HVX_ਸਦਿਸ਼ s4 = hvx_vec_splat_f32(2.642186986152672e-06f);
    HVX_ਸਦਿਸ਼ s3 = hvx_vec_splat_f32(-0.00019825318964070864f);
    HVX_ਸਦਿਸ਼ s2 = hvx_vec_splat_f32(0.00833326283319605f);
    HVX_ਸਦਿਸ਼ s1 = hvx_vec_splat_f32(-0.16666666082087775f);
    HVX_ਸਦਿਸ਼ s0 = hvx_vec_splat_f32(0.999999999915155f);

    HVX_ਸਦਿਸ਼ sin_y = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(s3, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(z, s4))));
    sin_y = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(s2, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(z, sin_y))));
    sin_y = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(s1, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(z, sin_y))));
    sin_y = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(s0, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(z, sin_y))));
    sin_y = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(y, sin_y));

    *vcos = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(cos_y, sign));
    *vsin = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(sin_y, sign));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_cos_f32(HVX_ਸਦਿਸ਼ x) {
    HVX_ਸਦਿਸ਼ vcos;
    HVX_ਸਦਿਸ਼ vsin;
    hvx_vec_sincos_f32(x, &vcos, &vsin);
    ਵਾਪਸ vcos;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_sin_f32(HVX_ਸਦਿਸ਼ x) {
    HVX_ਸਦਿਸ਼ vcos;
    HVX_ਸਦਿਸ਼ vsin;
    hvx_vec_sincos_f32(x, &vcos, &vsin);
    ਵਾਪਸ vsin;
}

#ਜੇ_ਅੰਤ /* HVX_SIN_COS_H */
