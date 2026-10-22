#اگر_غیر_تعریف_شدہ HVX_NORM_H
#تعریف_کرو HVX_NORM_H

#شامل <stdint.h>
#شامل "hvx-base.h"
#شامل "hvx-reduce.h"
#شامل "hvx-inverse.h"
#شامل "hvx-sqrt.h"
#شامل "hvx-repl.h"

ساکن اندرونی خالی hvx_fast_rms_norm_f32(ثابت uint8_t * محدود src,
                                         uint8_t * محدود dst,
                                         ثابت صحیح عدد_elems,
                                         اعشاری     epsilon) {

    ثابت HVX_سمتیہ * محدود v_src = (HVX_سمتیہ *) src;
    HVX_سمتیہ * محدود v_dst       = (HVX_سمتیہ *) dst;

    ثابت صحیح nvec = عدد_elems / VLEN_FP32;    // number of full vectors
    ثابت صحیح nloe = عدد_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares for full vectors
    HVX_سمتیہ مجموع_v = Q6_V_vsplat_R(0x00000000);
    HVX_سمتیہ epsilon_v = hvx_vec_splat_f32(epsilon);

    #ہدایت unroll(4)
    برائے (صحیح i = 0; i < nvec; i++) {
        HVX_سمتیہ v1 = v_src[i];
        HVX_سمتیہ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        مجموع_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموع_v, v2);
    }

    // Handle tail elements using vectorized ops with masking
    اگر (nloe > 0) {
        HVX_سمتیہ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_سمتیہ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_سمتیہ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        مجموع_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموع_v, v2);
    }

    // Reduce HVX sum
    مجموع_v = hvx_vec_reduce_مجموع_f32(Q6_Vsf_equals_Vqf32(مجموع_v));

    HVX_سمتیہ t_v            = hvx_vec_splat_f32((اعشاری) عدد_elems);
    HVX_سمتیہ denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_سمتیہ اوسط_v         = Q6_Vqf32_vmpy_VsfVsf(مجموع_v, denom_v);
    HVX_سمتیہ اوسط_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(اوسط_v, epsilon_v);

    // Scale full vectors
    HVX_سمتیہ scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(اوسط_epsilon_v));

    #ہدایت unroll(4)
    برائے (صحیح i = 0; i < nvec; i++) {
        HVX_سمتیہ v1 = v_src[i];
        HVX_سمتیہ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        v_dst[i] = Q6_Vsf_equals_Vqf32(v2);
    }

    // Handle tail elements using vectorized ops with masking
    اگر (nloe > 0) {
        HVX_سمتیہ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_سمتیہ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_سمتیہ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_سمتیہ نتیجہ = Q6_Vsf_equals_Vqf32(v2);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, نتیجہ);
    }
}

ساکن اندرونی خالی hvx_fast_rms_norm_mul_f32(ثابت uint8_t * محدود src,
                                             ثابت uint8_t * محدود وزن,
                                             uint8_t * محدود dst,
                                             ثابت صحیح عدد_elems,
                                             اعشاری     epsilon) {
    ثابت HVX_سمتیہ * محدود v_src    = (ثابت HVX_سمتیہ *) src;
    ثابت HVX_سمتیہ * محدود v_وزن = (ثابت HVX_سمتیہ *) وزن;
    HVX_سمتیہ * محدود v_dst          = (HVX_سمتیہ *) dst;

    ثابت صحیح nvec = عدد_elems / VLEN_FP32;    // number of full vectors
    ثابت صحیح nloe = عدد_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares for full vectors
    HVX_سمتیہ مجموع_v = Q6_V_vsplat_R(0x00000000);
    HVX_سمتیہ epsilon_v = hvx_vec_splat_f32(epsilon);

    #ہدایت unroll(4)
    برائے (صحیح i = 0; i < nvec; i++) {
        HVX_سمتیہ v1 = v_src[i];
        HVX_سمتیہ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        مجموع_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموع_v, v2);
    }

    // Handle tail elements using vectorized ops with masking
    اگر (nloe > 0) {
        HVX_سمتیہ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_سمتیہ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_سمتیہ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        مجموع_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموع_v, v2);
    }

    // Reduce HVX sum
    مجموع_v = hvx_vec_reduce_مجموع_f32(Q6_Vsf_equals_Vqf32(مجموع_v));

    HVX_سمتیہ t_v            = hvx_vec_splat_f32((اعشاری) عدد_elems);
    HVX_سمتیہ denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_سمتیہ اوسط_v         = Q6_Vqf32_vmpy_VsfVsf(مجموع_v, denom_v);
    HVX_سمتیہ اوسط_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(اوسط_v, epsilon_v);

    // Scale and multiply
    HVX_سمتیہ scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(اوسط_epsilon_v));

    #ہدایت unroll(4)
    برائے (صحیح i = 0; i < nvec; i++) {
        HVX_سمتیہ v1 = v_src[i];
        HVX_سمتیہ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_سمتیہ v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_سمتیہ نتیجہ = Q6_Vqf32_vmpy_VsfVsf(v3, v_وزن[i]);
        v_dst[i] = Q6_Vsf_equals_Vqf32(نتیجہ);
    }

    // Handle tail elements using vectorized ops with masking
    اگر (nloe > 0) {
        HVX_سمتیہ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_سمتیہ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_سمتیہ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_سمتیہ v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_سمتیہ نتیجہ = Q6_Vqf32_vmpy_VsfVsf(v3, v_وزن[nvec]);
        HVX_سمتیہ res_v = Q6_Vsf_equals_Vqf32(نتیجہ);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, res_v);
    }
}

ساکن اندرونی خالی hvx_fast_norm_f32(ثابت uint8_t * محدود src,
                                     uint8_t * محدود dst,
                                     ثابت صحیح عدد_elems,
                                     اعشاری     epsilon) {

    ثابت HVX_سمتیہ * محدود v_src = (HVX_سمتیہ *) src;
    HVX_سمتیہ * محدود v_dst       = (HVX_سمتیہ *) dst;

    ثابت صحیح nvec = عدد_elems / VLEN_FP32;    // number of full vectors
    ثابت صحیح nloe = عدد_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares and sum of values for full vectors
    HVX_سمتیہ مجموع_sq_v = Q6_V_vsplat_R(0x00000000);
    HVX_سمتیہ مجموع_x_v  = Q6_V_vsplat_R(0x00000000);
    HVX_سمتیہ epsilon_v = hvx_vec_splat_f32(epsilon);

    #ہدایت unroll(4)
    برائے (صحیح i = 0; i < nvec; i++) {
        HVX_سمتیہ v1 = v_src[i];
        HVX_سمتیہ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        مجموع_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموع_sq_v, v2);
        مجموع_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(مجموع_x_v,  Q6_Vqf32_vadd_VsfVsf(v1, Q6_V_vzero()));
    }

    // Handle tail elements using vectorized ops with masking
    اگر (nloe > 0) {
        HVX_سمتیہ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_سمتیہ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_سمتیہ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        مجموع_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموع_sq_v, v2);
        مجموع_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(مجموع_x_v,  Q6_Vqf32_vadd_VsfVsf(v1, Q6_V_vzero()));
    }

    // Reduce HVX sums
    مجموع_sq_v = hvx_vec_reduce_مجموع_f32(Q6_Vsf_equals_Vqf32(مجموع_sq_v));
    مجموع_x_v  = hvx_vec_reduce_مجموع_f32(Q6_Vsf_equals_Vqf32(مجموع_x_v));

    HVX_سمتیہ t_v            = hvx_vec_splat_f32((اعشاری) عدد_elems);
    HVX_سمتیہ denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_سمتیہ اوسط_sq_v      = Q6_Vqf32_vmpy_VsfVsf(مجموع_sq_v, denom_v);
    HVX_سمتیہ اوسط_x_v       = Q6_Vqf32_vmpy_VsfVsf(مجموع_x_v,  denom_v);
    HVX_سمتیہ اوسط_x_sq_v    = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(اوسط_x_v), Q6_Vsf_equals_Vqf32(اوسط_x_v));
    HVX_سمتیہ var_v          = Q6_Vqf32_vsub_Vqf32Vqf32(اوسط_sq_v, اوسط_x_sq_v);
    HVX_سمتیہ var_epsilon_v  = Q6_Vqf32_vadd_Vqf32Vsf(var_v, epsilon_v);

    // scale = rsqrt(variance + epsilon),  mean_x broadcast for subtraction
    HVX_سمتیہ scale_v  = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(var_epsilon_v));
    HVX_سمتیہ اوسط_x_b = hvx_vec_repl_f32(Q6_Vsf_equals_Vqf32(اوسط_x_v));

    #ہدایت unroll(4)
    برائے (صحیح i = 0; i < nvec; i++) {
        HVX_سمتیہ v1 = v_src[i];
        HVX_سمتیہ v2 = Q6_Vqf32_vsub_VsfVsf(v1, اوسط_x_b);
        HVX_سمتیہ v3 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v2), scale_v);
        v_dst[i] = Q6_Vsf_equals_Vqf32(v3);
    }

    // Handle tail elements using vectorized ops with masking
    اگر (nloe > 0) {
        HVX_سمتیہ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_سمتیہ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_سمتیہ v2 = Q6_Vqf32_vsub_VsfVsf(v1, اوسط_x_b);
        HVX_سمتیہ v3 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v2), scale_v);
        HVX_سمتیہ نتیجہ = Q6_Vsf_equals_Vqf32(v3);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, نتیجہ);
    }
}

ساکن اندرونی خالی hvx_fast_l2_norm_f32(ثابت uint8_t * محدود src,
                                        uint8_t * محدود dst,
                                        ثابت صحیح عدد_elems,
                                        اعشاری     epsilon) {

    ثابت HVX_سمتیہ * محدود v_src = (HVX_سمتیہ *) src;
    HVX_سمتیہ * محدود v_dst       = (HVX_سمتیہ *) dst;

    HVX_سمتیہ مجموع_v = hvx_vec_splat_f32(0.0f);

    ثابت صحیح nvec = عدد_elems / VLEN_FP32;
    ثابت صحیح nloe = عدد_elems % VLEN_FP32;

    #ہدایت unroll(4)
    برائے (صحیح i = 0; i < nvec; i++) {
        HVX_سمتیہ v1 = v_src[i];
        HVX_سمتیہ sq = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        مجموع_v         = Q6_Vqf32_vadd_Vqf32Vqf32(مجموع_v, sq);
    }

    // Include tail elements in the sum-of-squares using a predicate mask
    اگر (nloe > 0) {
        HVX_سمتیہ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_سمتیہ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_سمتیہ sq = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        مجموع_v         = Q6_Vqf32_vadd_Vqf32Vqf32(مجموع_v, sq);
    }

    // Compute scale = 1/fmax(sqrt(sum), epsilon) entirely in HVX registers.
    // hvx_vec_rsqrt_f32 + hvx_vec_inverse_f32 avoids scalar extraction.
    HVX_سمتیہ مجموع_sf    = hvx_vec_reduce_مجموع_f32(Q6_Vsf_equals_Vqf32(مجموع_v));
    HVX_سمتیہ rsqrt_v   = hvx_vec_rsqrt_f32(مجموع_sf);              // 1/sqrt(sum)
    HVX_سمتیہ sqrt_v    = hvx_vec_inverse_f32(rsqrt_v);            // sqrt(sum)
    HVX_سمتیہ epsilon_v = hvx_vec_splat_f32(epsilon);
    HVX_سمتیہ denom_v   = Q6_Vsf_vmax_VsfVsf(sqrt_v, epsilon_v);  // fmax(sqrt(sum), epsilon)
    HVX_سمتیہ scale_v   = hvx_vec_inverse_f32(denom_v);            // 1/fmax(sqrt(sum), epsilon)

    #ہدایت unroll(4)
    برائے (صحیح i = 0; i < nvec; i++) {
        HVX_سمتیہ v1 = v_src[i];
        v_dst[i]      = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, scale_v));
    }

    اگر (nloe > 0) {
        HVX_سمتیہ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_سمتیہ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_سمتیہ نتیجہ = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, scale_v));
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, نتیجہ);
    }
}

// F16 norm kernels: reduce and scale in f32 (via promote/narrow), matching the
// precision-preserving pattern used by the flash-attn f16 kernels.

ساکن اندرونی خالی hvx_fast_rms_norm_f16(ثابت uint8_t * محدود src,
                                         uint8_t * محدود dst,
                                         ثابت صحیح عدد_elems,
                                         اعشاری     epsilon) {

    ثابت HVX_سمتیہ * محدود v_src = (HVX_سمتیہ *) src;
    HVX_سمتیہ * محدود v_dst       = (HVX_سمتیہ *) dst;

    ثابت صحیح nvec = عدد_elems / VLEN_FP16;    // number of full f16 vectors
    ثابت صحیح nloe = عدد_elems % VLEN_FP16;    // leftover elements

    HVX_سمتیہ مجموع_v = Q6_V_vsplat_R(0x00000000);
    HVX_سمتیہ epsilon_v = hvx_vec_splat_f32(epsilon);

    #ہدایت unroll(4)
    برائے (صحیح i = 0; i < nvec; i++) {
        HVX_سمتیہ_Pair p = hvx_vec_f16_مقصد_f32(v_src[i]);
        HVX_سمتیہ p0 = Q6_V_lo_W(p);
        HVX_سمتیہ p1 = Q6_V_hi_W(p);
        مجموع_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموع_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        مجموع_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموع_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    اگر (nloe > 0) {
        HVX_سمتیہ_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_سمتیہ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_سمتیہ_Pair p = hvx_vec_f16_مقصد_f32(v1);
        HVX_سمتیہ p0 = Q6_V_lo_W(p);
        HVX_سمتیہ p1 = Q6_V_hi_W(p);
        مجموع_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموع_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        مجموع_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموع_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    مجموع_v = hvx_vec_reduce_مجموع_f32(Q6_Vsf_equals_Vqf32(مجموع_v));

    HVX_سمتیہ t_v            = hvx_vec_splat_f32((اعشاری) عدد_elems);
    HVX_سمتیہ denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_سمتیہ اوسط_v         = Q6_Vqf32_vmpy_VsfVsf(مجموع_v, denom_v);
    HVX_سمتیہ اوسط_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(اوسط_v, epsilon_v);

    HVX_سمتیہ scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(اوسط_epsilon_v));

    #ہدایت unroll(4)
    برائے (صحیح i = 0; i < nvec; i++) {
        HVX_سمتیہ_Pair p = hvx_vec_f16_مقصد_f32(v_src[i]);
        HVX_سمتیہ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_سمتیہ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        v_dst[i] = hvx_vec_f32_مقصد_f16(r0, r1);
    }

    اگر (nloe > 0) {
        HVX_سمتیہ_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_سمتیہ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_سمتیہ_Pair p = hvx_vec_f16_مقصد_f32(v1);
        HVX_سمتیہ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_سمتیہ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        HVX_سمتیہ نتیجہ = hvx_vec_f32_مقصد_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, نتیجہ);
    }
}

ساکن اندرونی خالی hvx_fast_norm_f16(ثابت uint8_t * محدود src,
                                     uint8_t * محدود dst,
                                     ثابت صحیح عدد_elems,
                                     اعشاری     epsilon) {

    ثابت HVX_سمتیہ * محدود v_src = (HVX_سمتیہ *) src;
    HVX_سمتیہ * محدود v_dst       = (HVX_سمتیہ *) dst;

    ثابت صحیح nvec = عدد_elems / VLEN_FP16;
    ثابت صحیح nloe = عدد_elems % VLEN_FP16;

    HVX_سمتیہ مجموع_sq_v = Q6_V_vsplat_R(0x00000000);
    HVX_سمتیہ مجموع_x_v  = Q6_V_vsplat_R(0x00000000);
    HVX_سمتیہ epsilon_v = hvx_vec_splat_f32(epsilon);

    #ہدایت unroll(4)
    برائے (صحیح i = 0; i < nvec; i++) {
        HVX_سمتیہ_Pair p = hvx_vec_f16_مقصد_f32(v_src[i]);
        HVX_سمتیہ p0 = Q6_V_lo_W(p);
        HVX_سمتیہ p1 = Q6_V_hi_W(p);
        مجموع_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموع_sq_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        مجموع_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموع_sq_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
        مجموع_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(مجموع_x_v,  Q6_Vqf32_vadd_VsfVsf(p0, Q6_V_vzero()));
        مجموع_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(مجموع_x_v,  Q6_Vqf32_vadd_VsfVsf(p1, Q6_V_vzero()));
    }

    اگر (nloe > 0) {
        HVX_سمتیہ_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_سمتیہ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_سمتیہ_Pair p = hvx_vec_f16_مقصد_f32(v1);
        HVX_سمتیہ p0 = Q6_V_lo_W(p);
        HVX_سمتیہ p1 = Q6_V_hi_W(p);
        مجموع_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموع_sq_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        مجموع_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموع_sq_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
        مجموع_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(مجموع_x_v,  Q6_Vqf32_vadd_VsfVsf(p0, Q6_V_vzero()));
        مجموع_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(مجموع_x_v,  Q6_Vqf32_vadd_VsfVsf(p1, Q6_V_vzero()));
    }

    مجموع_sq_v = hvx_vec_reduce_مجموع_f32(Q6_Vsf_equals_Vqf32(مجموع_sq_v));
    مجموع_x_v  = hvx_vec_reduce_مجموع_f32(Q6_Vsf_equals_Vqf32(مجموع_x_v));

    HVX_سمتیہ t_v            = hvx_vec_splat_f32((اعشاری) عدد_elems);
    HVX_سمتیہ denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_سمتیہ اوسط_sq_v      = Q6_Vqf32_vmpy_VsfVsf(مجموع_sq_v, denom_v);
    HVX_سمتیہ اوسط_x_v       = Q6_Vqf32_vmpy_VsfVsf(مجموع_x_v,  denom_v);
    HVX_سمتیہ اوسط_x_sq_v    = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(اوسط_x_v), Q6_Vsf_equals_Vqf32(اوسط_x_v));
    HVX_سمتیہ var_v          = Q6_Vqf32_vsub_Vqf32Vqf32(اوسط_sq_v, اوسط_x_sq_v);
    HVX_سمتیہ var_epsilon_v  = Q6_Vqf32_vadd_Vqf32Vsf(var_v, epsilon_v);

    HVX_سمتیہ scale_v  = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(var_epsilon_v));
    HVX_سمتیہ اوسط_x_b = hvx_vec_repl_f32(Q6_Vsf_equals_Vqf32(اوسط_x_v));

    #ہدایت unroll(4)
    برائے (صحیح i = 0; i < nvec; i++) {
        HVX_سمتیہ_Pair p = hvx_vec_f16_مقصد_f32(v_src[i]);
        HVX_سمتیہ d0 = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(p), اوسط_x_b);
        HVX_سمتیہ d1 = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(p), اوسط_x_b);
        HVX_سمتیہ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d0), scale_v));
        HVX_سمتیہ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d1), scale_v));
        v_dst[i] = hvx_vec_f32_مقصد_f16(r0, r1);
    }

    اگر (nloe > 0) {
        HVX_سمتیہ_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_سمتیہ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_سمتیہ_Pair p = hvx_vec_f16_مقصد_f32(v1);
        HVX_سمتیہ d0 = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(p), اوسط_x_b);
        HVX_سمتیہ d1 = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(p), اوسط_x_b);
        HVX_سمتیہ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d0), scale_v));
        HVX_سمتیہ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d1), scale_v));
        HVX_سمتیہ نتیجہ = hvx_vec_f32_مقصد_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, نتیجہ);
    }
}

ساکن اندرونی خالی hvx_fast_l2_norm_f16(ثابت uint8_t * محدود src,
                                        uint8_t * محدود dst,
                                        ثابت صحیح عدد_elems,
                                        اعشاری     epsilon) {

    ثابت HVX_سمتیہ * محدود v_src = (HVX_سمتیہ *) src;
    HVX_سمتیہ * محدود v_dst       = (HVX_سمتیہ *) dst;

    ثابت صحیح nvec = عدد_elems / VLEN_FP16;
    ثابت صحیح nloe = عدد_elems % VLEN_FP16;

    HVX_سمتیہ مجموع_v = hvx_vec_splat_f32(0.0f);

    #ہدایت unroll(4)
    برائے (صحیح i = 0; i < nvec; i++) {
        HVX_سمتیہ_Pair p = hvx_vec_f16_مقصد_f32(v_src[i]);
        HVX_سمتیہ p0 = Q6_V_lo_W(p);
        HVX_سمتیہ p1 = Q6_V_hi_W(p);
        مجموع_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموع_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        مجموع_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموع_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    اگر (nloe > 0) {
        HVX_سمتیہ_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_سمتیہ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_سمتیہ_Pair p = hvx_vec_f16_مقصد_f32(v1);
        HVX_سمتیہ p0 = Q6_V_lo_W(p);
        HVX_سمتیہ p1 = Q6_V_hi_W(p);
        مجموع_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموع_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        مجموع_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموع_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    HVX_سمتیہ مجموع_sf    = hvx_vec_reduce_مجموع_f32(Q6_Vsf_equals_Vqf32(مجموع_v));
    HVX_سمتیہ rsqrt_v   = hvx_vec_rsqrt_f32(مجموع_sf);
    HVX_سمتیہ sqrt_v    = hvx_vec_inverse_f32(rsqrt_v);
    HVX_سمتیہ epsilon_v = hvx_vec_splat_f32(epsilon);
    HVX_سمتیہ denom_v   = Q6_Vsf_vmax_VsfVsf(sqrt_v, epsilon_v);
    HVX_سمتیہ scale_v   = hvx_vec_inverse_f32(denom_v);

    #ہدایت unroll(4)
    برائے (صحیح i = 0; i < nvec; i++) {
        HVX_سمتیہ_Pair p = hvx_vec_f16_مقصد_f32(v_src[i]);
        HVX_سمتیہ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_سمتیہ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        v_dst[i] = hvx_vec_f32_مقصد_f16(r0, r1);
    }

    اگر (nloe > 0) {
        HVX_سمتیہ_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_سمتیہ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_سمتیہ_Pair p = hvx_vec_f16_مقصد_f32(v1);
        HVX_سمتیہ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_سمتیہ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        HVX_سمتیہ نتیجہ = hvx_vec_f32_مقصد_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, نتیجہ);
    }
}

#اگر_ختم // HVX_NORM_H
