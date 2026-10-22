#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HVX_NORM_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_NORM_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-base.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-reduce.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-inverse.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-sqrt.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-repl.h"

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_fast_rms_norm_f32(ਅਚਲ uint8_t * ਸੀਮਿਤ src,
                                         uint8_t * ਸੀਮਿਤ dst,
                                         ਅਚਲ ਪੂਰਨ ਅੰਕ_elems,
                                         ਦਸ਼ਮਲਵ     epsilon) {

    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_src = (HVX_ਸਦਿਸ਼ *) src;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_dst       = (HVX_ਸਦਿਸ਼ *) dst;

    ਅਚਲ ਪੂਰਨ nvec = ਅੰਕ_elems / VLEN_FP32;    // number of full vectors
    ਅਚਲ ਪੂਰਨ nloe = ਅੰਕ_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares for full vectors
    HVX_ਸਦਿਸ਼ ਜੋੜ_v = Q6_V_vsplat_R(0x00000000);
    HVX_ਸਦਿਸ਼ epsilon_v = hvx_vec_splat_f32(epsilon);

    #ਹਦਾਇਤ unroll(4)
    ਲਈ (ਪੂਰਨ i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ v1 = v_src[i];
        HVX_ਸਦਿਸ਼ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        ਜੋੜ_v = Q6_Vqf32_vadd_Vqf32Vqf32(ਜੋੜ_v, v2);
    }

    // Handle tail elements using vectorized ops with masking
    ਜੇ (nloe > 0) {
        HVX_ਸਦਿਸ਼_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ਸਦਿਸ਼ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ਸਦਿਸ਼ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        ਜੋੜ_v = Q6_Vqf32_vadd_Vqf32Vqf32(ਜੋੜ_v, v2);
    }

    // Reduce HVX sum
    ਜੋੜ_v = hvx_vec_reduce_ਜੋੜ_f32(Q6_Vsf_equals_Vqf32(ਜੋੜ_v));

    HVX_ਸਦਿਸ਼ t_v            = hvx_vec_splat_f32((ਦਸ਼ਮਲਵ) ਅੰਕ_elems);
    HVX_ਸਦਿਸ਼ denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_ਸਦਿਸ਼ ਔਸਤ_v         = Q6_Vqf32_vmpy_VsfVsf(ਜੋੜ_v, denom_v);
    HVX_ਸਦਿਸ਼ ਔਸਤ_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(ਔਸਤ_v, epsilon_v);

    // Scale full vectors
    HVX_ਸਦਿਸ਼ scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(ਔਸਤ_epsilon_v));

    #ਹਦਾਇਤ unroll(4)
    ਲਈ (ਪੂਰਨ i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ v1 = v_src[i];
        HVX_ਸਦਿਸ਼ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        v_dst[i] = Q6_Vsf_equals_Vqf32(v2);
    }

    // Handle tail elements using vectorized ops with masking
    ਜੇ (nloe > 0) {
        HVX_ਸਦਿਸ਼_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ਸਦਿਸ਼ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ਸਦਿਸ਼ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_ਸਦਿਸ਼ ਨਤੀਜਾ = Q6_Vsf_equals_Vqf32(v2);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, ਨਤੀਜਾ);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_fast_rms_norm_mul_f32(ਅਚਲ uint8_t * ਸੀਮਿਤ src,
                                             ਅਚਲ uint8_t * ਸੀਮਿਤ ਭਾਰ,
                                             uint8_t * ਸੀਮਿਤ dst,
                                             ਅਚਲ ਪੂਰਨ ਅੰਕ_elems,
                                             ਦਸ਼ਮਲਵ     epsilon) {
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_src    = (ਅਚਲ HVX_ਸਦਿਸ਼ *) src;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_ਭਾਰ = (ਅਚਲ HVX_ਸਦਿਸ਼ *) ਭਾਰ;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_dst          = (HVX_ਸਦਿਸ਼ *) dst;

    ਅਚਲ ਪੂਰਨ nvec = ਅੰਕ_elems / VLEN_FP32;    // number of full vectors
    ਅਚਲ ਪੂਰਨ nloe = ਅੰਕ_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares for full vectors
    HVX_ਸਦਿਸ਼ ਜੋੜ_v = Q6_V_vsplat_R(0x00000000);
    HVX_ਸਦਿਸ਼ epsilon_v = hvx_vec_splat_f32(epsilon);

    #ਹਦਾਇਤ unroll(4)
    ਲਈ (ਪੂਰਨ i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ v1 = v_src[i];
        HVX_ਸਦਿਸ਼ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        ਜੋੜ_v = Q6_Vqf32_vadd_Vqf32Vqf32(ਜੋੜ_v, v2);
    }

    // Handle tail elements using vectorized ops with masking
    ਜੇ (nloe > 0) {
        HVX_ਸਦਿਸ਼_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ਸਦਿਸ਼ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ਸਦਿਸ਼ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        ਜੋੜ_v = Q6_Vqf32_vadd_Vqf32Vqf32(ਜੋੜ_v, v2);
    }

    // Reduce HVX sum
    ਜੋੜ_v = hvx_vec_reduce_ਜੋੜ_f32(Q6_Vsf_equals_Vqf32(ਜੋੜ_v));

    HVX_ਸਦਿਸ਼ t_v            = hvx_vec_splat_f32((ਦਸ਼ਮਲਵ) ਅੰਕ_elems);
    HVX_ਸਦਿਸ਼ denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_ਸਦਿਸ਼ ਔਸਤ_v         = Q6_Vqf32_vmpy_VsfVsf(ਜੋੜ_v, denom_v);
    HVX_ਸਦਿਸ਼ ਔਸਤ_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(ਔਸਤ_v, epsilon_v);

    // Scale and multiply
    HVX_ਸਦਿਸ਼ scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(ਔਸਤ_epsilon_v));

    #ਹਦਾਇਤ unroll(4)
    ਲਈ (ਪੂਰਨ i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ v1 = v_src[i];
        HVX_ਸਦਿਸ਼ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_ਸਦਿਸ਼ v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_ਸਦਿਸ਼ ਨਤੀਜਾ = Q6_Vqf32_vmpy_VsfVsf(v3, v_ਭਾਰ[i]);
        v_dst[i] = Q6_Vsf_equals_Vqf32(ਨਤੀਜਾ);
    }

    // Handle tail elements using vectorized ops with masking
    ਜੇ (nloe > 0) {
        HVX_ਸਦਿਸ਼_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ਸਦਿਸ਼ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ਸਦਿਸ਼ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_ਸਦਿਸ਼ v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_ਸਦਿਸ਼ ਨਤੀਜਾ = Q6_Vqf32_vmpy_VsfVsf(v3, v_ਭਾਰ[nvec]);
        HVX_ਸਦਿਸ਼ res_v = Q6_Vsf_equals_Vqf32(ਨਤੀਜਾ);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, res_v);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_fast_norm_f32(ਅਚਲ uint8_t * ਸੀਮਿਤ src,
                                     uint8_t * ਸੀਮਿਤ dst,
                                     ਅਚਲ ਪੂਰਨ ਅੰਕ_elems,
                                     ਦਸ਼ਮਲਵ     epsilon) {

    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_src = (HVX_ਸਦਿਸ਼ *) src;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_dst       = (HVX_ਸਦਿਸ਼ *) dst;

    ਅਚਲ ਪੂਰਨ nvec = ਅੰਕ_elems / VLEN_FP32;    // number of full vectors
    ਅਚਲ ਪੂਰਨ nloe = ਅੰਕ_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares and sum of values for full vectors
    HVX_ਸਦਿਸ਼ ਜੋੜ_sq_v = Q6_V_vsplat_R(0x00000000);
    HVX_ਸਦਿਸ਼ ਜੋੜ_x_v  = Q6_V_vsplat_R(0x00000000);
    HVX_ਸਦਿਸ਼ epsilon_v = hvx_vec_splat_f32(epsilon);

    #ਹਦਾਇਤ unroll(4)
    ਲਈ (ਪੂਰਨ i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ v1 = v_src[i];
        HVX_ਸਦਿਸ਼ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        ਜੋੜ_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(ਜੋੜ_sq_v, v2);
        ਜੋੜ_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(ਜੋੜ_x_v,  Q6_Vqf32_vadd_VsfVsf(v1, Q6_V_vzero()));
    }

    // Handle tail elements using vectorized ops with masking
    ਜੇ (nloe > 0) {
        HVX_ਸਦਿਸ਼_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ਸਦਿਸ਼ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ਸਦਿਸ਼ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        ਜੋੜ_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(ਜੋੜ_sq_v, v2);
        ਜੋੜ_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(ਜੋੜ_x_v,  Q6_Vqf32_vadd_VsfVsf(v1, Q6_V_vzero()));
    }

    // Reduce HVX sums
    ਜੋੜ_sq_v = hvx_vec_reduce_ਜੋੜ_f32(Q6_Vsf_equals_Vqf32(ਜੋੜ_sq_v));
    ਜੋੜ_x_v  = hvx_vec_reduce_ਜੋੜ_f32(Q6_Vsf_equals_Vqf32(ਜੋੜ_x_v));

    HVX_ਸਦਿਸ਼ t_v            = hvx_vec_splat_f32((ਦਸ਼ਮਲਵ) ਅੰਕ_elems);
    HVX_ਸਦਿਸ਼ denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_ਸਦਿਸ਼ ਔਸਤ_sq_v      = Q6_Vqf32_vmpy_VsfVsf(ਜੋੜ_sq_v, denom_v);
    HVX_ਸਦਿਸ਼ ਔਸਤ_x_v       = Q6_Vqf32_vmpy_VsfVsf(ਜੋੜ_x_v,  denom_v);
    HVX_ਸਦਿਸ਼ ਔਸਤ_x_sq_v    = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(ਔਸਤ_x_v), Q6_Vsf_equals_Vqf32(ਔਸਤ_x_v));
    HVX_ਸਦਿਸ਼ var_v          = Q6_Vqf32_vsub_Vqf32Vqf32(ਔਸਤ_sq_v, ਔਸਤ_x_sq_v);
    HVX_ਸਦਿਸ਼ var_epsilon_v  = Q6_Vqf32_vadd_Vqf32Vsf(var_v, epsilon_v);

    // scale = rsqrt(variance + epsilon),  mean_x broadcast for subtraction
    HVX_ਸਦਿਸ਼ scale_v  = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(var_epsilon_v));
    HVX_ਸਦਿਸ਼ ਔਸਤ_x_b = hvx_vec_repl_f32(Q6_Vsf_equals_Vqf32(ਔਸਤ_x_v));

    #ਹਦਾਇਤ unroll(4)
    ਲਈ (ਪੂਰਨ i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ v1 = v_src[i];
        HVX_ਸਦਿਸ਼ v2 = Q6_Vqf32_vsub_VsfVsf(v1, ਔਸਤ_x_b);
        HVX_ਸਦਿਸ਼ v3 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v2), scale_v);
        v_dst[i] = Q6_Vsf_equals_Vqf32(v3);
    }

    // Handle tail elements using vectorized ops with masking
    ਜੇ (nloe > 0) {
        HVX_ਸਦਿਸ਼_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ਸਦਿਸ਼ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ਸਦਿਸ਼ v2 = Q6_Vqf32_vsub_VsfVsf(v1, ਔਸਤ_x_b);
        HVX_ਸਦਿਸ਼ v3 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v2), scale_v);
        HVX_ਸਦਿਸ਼ ਨਤੀਜਾ = Q6_Vsf_equals_Vqf32(v3);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, ਨਤੀਜਾ);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_fast_l2_norm_f32(ਅਚਲ uint8_t * ਸੀਮਿਤ src,
                                        uint8_t * ਸੀਮਿਤ dst,
                                        ਅਚਲ ਪੂਰਨ ਅੰਕ_elems,
                                        ਦਸ਼ਮਲਵ     epsilon) {

    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_src = (HVX_ਸਦਿਸ਼ *) src;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_dst       = (HVX_ਸਦਿਸ਼ *) dst;

    HVX_ਸਦਿਸ਼ ਜੋੜ_v = hvx_vec_splat_f32(0.0f);

    ਅਚਲ ਪੂਰਨ nvec = ਅੰਕ_elems / VLEN_FP32;
    ਅਚਲ ਪੂਰਨ nloe = ਅੰਕ_elems % VLEN_FP32;

    #ਹਦਾਇਤ unroll(4)
    ਲਈ (ਪੂਰਨ i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ v1 = v_src[i];
        HVX_ਸਦਿਸ਼ sq = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        ਜੋੜ_v         = Q6_Vqf32_vadd_Vqf32Vqf32(ਜੋੜ_v, sq);
    }

    // Include tail elements in the sum-of-squares using a predicate mask
    ਜੇ (nloe > 0) {
        HVX_ਸਦਿਸ਼_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ਸਦਿਸ਼ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ਸਦਿਸ਼ sq = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        ਜੋੜ_v         = Q6_Vqf32_vadd_Vqf32Vqf32(ਜੋੜ_v, sq);
    }

    // Compute scale = 1/fmax(sqrt(sum), epsilon) entirely in HVX registers.
    // hvx_vec_rsqrt_f32 + hvx_vec_inverse_f32 avoids scalar extraction.
    HVX_ਸਦਿਸ਼ ਜੋੜ_sf    = hvx_vec_reduce_ਜੋੜ_f32(Q6_Vsf_equals_Vqf32(ਜੋੜ_v));
    HVX_ਸਦਿਸ਼ rsqrt_v   = hvx_vec_rsqrt_f32(ਜੋੜ_sf);              // 1/sqrt(sum)
    HVX_ਸਦਿਸ਼ sqrt_v    = hvx_vec_inverse_f32(rsqrt_v);            // sqrt(sum)
    HVX_ਸਦਿਸ਼ epsilon_v = hvx_vec_splat_f32(epsilon);
    HVX_ਸਦਿਸ਼ denom_v   = Q6_Vsf_vmax_VsfVsf(sqrt_v, epsilon_v);  // fmax(sqrt(sum), epsilon)
    HVX_ਸਦਿਸ਼ scale_v   = hvx_vec_inverse_f32(denom_v);            // 1/fmax(sqrt(sum), epsilon)

    #ਹਦਾਇਤ unroll(4)
    ਲਈ (ਪੂਰਨ i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ v1 = v_src[i];
        v_dst[i]      = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, scale_v));
    }

    ਜੇ (nloe > 0) {
        HVX_ਸਦਿਸ਼_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ਸਦਿਸ਼ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ਸਦਿਸ਼ ਨਤੀਜਾ = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, scale_v));
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, ਨਤੀਜਾ);
    }
}

// F16 norm kernels: reduce and scale in f32 (via promote/narrow), matching the
// precision-preserving pattern used by the flash-attn f16 kernels.

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_fast_rms_norm_f16(ਅਚਲ uint8_t * ਸੀਮਿਤ src,
                                         uint8_t * ਸੀਮਿਤ dst,
                                         ਅਚਲ ਪੂਰਨ ਅੰਕ_elems,
                                         ਦਸ਼ਮਲਵ     epsilon) {

    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_src = (HVX_ਸਦਿਸ਼ *) src;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_dst       = (HVX_ਸਦਿਸ਼ *) dst;

    ਅਚਲ ਪੂਰਨ nvec = ਅੰਕ_elems / VLEN_FP16;    // number of full f16 vectors
    ਅਚਲ ਪੂਰਨ nloe = ਅੰਕ_elems % VLEN_FP16;    // leftover elements

    HVX_ਸਦਿਸ਼ ਜੋੜ_v = Q6_V_vsplat_R(0x00000000);
    HVX_ਸਦਿਸ਼ epsilon_v = hvx_vec_splat_f32(epsilon);

    #ਹਦਾਇਤ unroll(4)
    ਲਈ (ਪੂਰਨ i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼_Pair p = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(v_src[i]);
        HVX_ਸਦਿਸ਼ p0 = Q6_V_lo_W(p);
        HVX_ਸਦਿਸ਼ p1 = Q6_V_hi_W(p);
        ਜੋੜ_v = Q6_Vqf32_vadd_Vqf32Vqf32(ਜੋੜ_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        ਜੋੜ_v = Q6_Vqf32_vadd_Vqf32Vqf32(ਜੋੜ_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    ਜੇ (nloe > 0) {
        HVX_ਸਦਿਸ਼_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_ਸਦਿਸ਼ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ਸਦਿਸ਼_Pair p = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(v1);
        HVX_ਸਦਿਸ਼ p0 = Q6_V_lo_W(p);
        HVX_ਸਦਿਸ਼ p1 = Q6_V_hi_W(p);
        ਜੋੜ_v = Q6_Vqf32_vadd_Vqf32Vqf32(ਜੋੜ_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        ਜੋੜ_v = Q6_Vqf32_vadd_Vqf32Vqf32(ਜੋੜ_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    ਜੋੜ_v = hvx_vec_reduce_ਜੋੜ_f32(Q6_Vsf_equals_Vqf32(ਜੋੜ_v));

    HVX_ਸਦਿਸ਼ t_v            = hvx_vec_splat_f32((ਦਸ਼ਮਲਵ) ਅੰਕ_elems);
    HVX_ਸਦਿਸ਼ denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_ਸਦਿਸ਼ ਔਸਤ_v         = Q6_Vqf32_vmpy_VsfVsf(ਜੋੜ_v, denom_v);
    HVX_ਸਦਿਸ਼ ਔਸਤ_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(ਔਸਤ_v, epsilon_v);

    HVX_ਸਦਿਸ਼ scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(ਔਸਤ_epsilon_v));

    #ਹਦਾਇਤ unroll(4)
    ਲਈ (ਪੂਰਨ i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼_Pair p = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(v_src[i]);
        HVX_ਸਦਿਸ਼ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_ਸਦਿਸ਼ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        v_dst[i] = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(r0, r1);
    }

    ਜੇ (nloe > 0) {
        HVX_ਸਦਿਸ਼_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_ਸਦਿਸ਼ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ਸਦਿਸ਼_Pair p = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(v1);
        HVX_ਸਦਿਸ਼ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_ਸਦਿਸ਼ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        HVX_ਸਦਿਸ਼ ਨਤੀਜਾ = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, ਨਤੀਜਾ);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_fast_norm_f16(ਅਚਲ uint8_t * ਸੀਮਿਤ src,
                                     uint8_t * ਸੀਮਿਤ dst,
                                     ਅਚਲ ਪੂਰਨ ਅੰਕ_elems,
                                     ਦਸ਼ਮਲਵ     epsilon) {

    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_src = (HVX_ਸਦਿਸ਼ *) src;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_dst       = (HVX_ਸਦਿਸ਼ *) dst;

    ਅਚਲ ਪੂਰਨ nvec = ਅੰਕ_elems / VLEN_FP16;
    ਅਚਲ ਪੂਰਨ nloe = ਅੰਕ_elems % VLEN_FP16;

    HVX_ਸਦਿਸ਼ ਜੋੜ_sq_v = Q6_V_vsplat_R(0x00000000);
    HVX_ਸਦਿਸ਼ ਜੋੜ_x_v  = Q6_V_vsplat_R(0x00000000);
    HVX_ਸਦਿਸ਼ epsilon_v = hvx_vec_splat_f32(epsilon);

    #ਹਦਾਇਤ unroll(4)
    ਲਈ (ਪੂਰਨ i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼_Pair p = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(v_src[i]);
        HVX_ਸਦਿਸ਼ p0 = Q6_V_lo_W(p);
        HVX_ਸਦਿਸ਼ p1 = Q6_V_hi_W(p);
        ਜੋੜ_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(ਜੋੜ_sq_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        ਜੋੜ_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(ਜੋੜ_sq_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
        ਜੋੜ_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(ਜੋੜ_x_v,  Q6_Vqf32_vadd_VsfVsf(p0, Q6_V_vzero()));
        ਜੋੜ_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(ਜੋੜ_x_v,  Q6_Vqf32_vadd_VsfVsf(p1, Q6_V_vzero()));
    }

    ਜੇ (nloe > 0) {
        HVX_ਸਦਿਸ਼_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_ਸਦਿਸ਼ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ਸਦਿਸ਼_Pair p = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(v1);
        HVX_ਸਦਿਸ਼ p0 = Q6_V_lo_W(p);
        HVX_ਸਦਿਸ਼ p1 = Q6_V_hi_W(p);
        ਜੋੜ_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(ਜੋੜ_sq_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        ਜੋੜ_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(ਜੋੜ_sq_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
        ਜੋੜ_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(ਜੋੜ_x_v,  Q6_Vqf32_vadd_VsfVsf(p0, Q6_V_vzero()));
        ਜੋੜ_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(ਜੋੜ_x_v,  Q6_Vqf32_vadd_VsfVsf(p1, Q6_V_vzero()));
    }

    ਜੋੜ_sq_v = hvx_vec_reduce_ਜੋੜ_f32(Q6_Vsf_equals_Vqf32(ਜੋੜ_sq_v));
    ਜੋੜ_x_v  = hvx_vec_reduce_ਜੋੜ_f32(Q6_Vsf_equals_Vqf32(ਜੋੜ_x_v));

    HVX_ਸਦਿਸ਼ t_v            = hvx_vec_splat_f32((ਦਸ਼ਮਲਵ) ਅੰਕ_elems);
    HVX_ਸਦਿਸ਼ denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_ਸਦਿਸ਼ ਔਸਤ_sq_v      = Q6_Vqf32_vmpy_VsfVsf(ਜੋੜ_sq_v, denom_v);
    HVX_ਸਦਿਸ਼ ਔਸਤ_x_v       = Q6_Vqf32_vmpy_VsfVsf(ਜੋੜ_x_v,  denom_v);
    HVX_ਸਦਿਸ਼ ਔਸਤ_x_sq_v    = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(ਔਸਤ_x_v), Q6_Vsf_equals_Vqf32(ਔਸਤ_x_v));
    HVX_ਸਦਿਸ਼ var_v          = Q6_Vqf32_vsub_Vqf32Vqf32(ਔਸਤ_sq_v, ਔਸਤ_x_sq_v);
    HVX_ਸਦਿਸ਼ var_epsilon_v  = Q6_Vqf32_vadd_Vqf32Vsf(var_v, epsilon_v);

    HVX_ਸਦਿਸ਼ scale_v  = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(var_epsilon_v));
    HVX_ਸਦਿਸ਼ ਔਸਤ_x_b = hvx_vec_repl_f32(Q6_Vsf_equals_Vqf32(ਔਸਤ_x_v));

    #ਹਦਾਇਤ unroll(4)
    ਲਈ (ਪੂਰਨ i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼_Pair p = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(v_src[i]);
        HVX_ਸਦਿਸ਼ d0 = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(p), ਔਸਤ_x_b);
        HVX_ਸਦਿਸ਼ d1 = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(p), ਔਸਤ_x_b);
        HVX_ਸਦਿਸ਼ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d0), scale_v));
        HVX_ਸਦਿਸ਼ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d1), scale_v));
        v_dst[i] = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(r0, r1);
    }

    ਜੇ (nloe > 0) {
        HVX_ਸਦਿਸ਼_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_ਸਦਿਸ਼ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ਸਦਿਸ਼_Pair p = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(v1);
        HVX_ਸਦਿਸ਼ d0 = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(p), ਔਸਤ_x_b);
        HVX_ਸਦਿਸ਼ d1 = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(p), ਔਸਤ_x_b);
        HVX_ਸਦਿਸ਼ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d0), scale_v));
        HVX_ਸਦਿਸ਼ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d1), scale_v));
        HVX_ਸਦਿਸ਼ ਨਤੀਜਾ = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, ਨਤੀਜਾ);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_fast_l2_norm_f16(ਅਚਲ uint8_t * ਸੀਮਿਤ src,
                                        uint8_t * ਸੀਮਿਤ dst,
                                        ਅਚਲ ਪੂਰਨ ਅੰਕ_elems,
                                        ਦਸ਼ਮਲਵ     epsilon) {

    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_src = (HVX_ਸਦਿਸ਼ *) src;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_dst       = (HVX_ਸਦਿਸ਼ *) dst;

    ਅਚਲ ਪੂਰਨ nvec = ਅੰਕ_elems / VLEN_FP16;
    ਅਚਲ ਪੂਰਨ nloe = ਅੰਕ_elems % VLEN_FP16;

    HVX_ਸਦਿਸ਼ ਜੋੜ_v = hvx_vec_splat_f32(0.0f);

    #ਹਦਾਇਤ unroll(4)
    ਲਈ (ਪੂਰਨ i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼_Pair p = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(v_src[i]);
        HVX_ਸਦਿਸ਼ p0 = Q6_V_lo_W(p);
        HVX_ਸਦਿਸ਼ p1 = Q6_V_hi_W(p);
        ਜੋੜ_v = Q6_Vqf32_vadd_Vqf32Vqf32(ਜੋੜ_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        ਜੋੜ_v = Q6_Vqf32_vadd_Vqf32Vqf32(ਜੋੜ_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    ਜੇ (nloe > 0) {
        HVX_ਸਦਿਸ਼_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_ਸਦਿਸ਼ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ਸਦਿਸ਼_Pair p = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(v1);
        HVX_ਸਦਿਸ਼ p0 = Q6_V_lo_W(p);
        HVX_ਸਦਿਸ਼ p1 = Q6_V_hi_W(p);
        ਜੋੜ_v = Q6_Vqf32_vadd_Vqf32Vqf32(ਜੋੜ_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        ਜੋੜ_v = Q6_Vqf32_vadd_Vqf32Vqf32(ਜੋੜ_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    HVX_ਸਦਿਸ਼ ਜੋੜ_sf    = hvx_vec_reduce_ਜੋੜ_f32(Q6_Vsf_equals_Vqf32(ਜੋੜ_v));
    HVX_ਸਦਿਸ਼ rsqrt_v   = hvx_vec_rsqrt_f32(ਜੋੜ_sf);
    HVX_ਸਦਿਸ਼ sqrt_v    = hvx_vec_inverse_f32(rsqrt_v);
    HVX_ਸਦਿਸ਼ epsilon_v = hvx_vec_splat_f32(epsilon);
    HVX_ਸਦਿਸ਼ denom_v   = Q6_Vsf_vmax_VsfVsf(sqrt_v, epsilon_v);
    HVX_ਸਦਿਸ਼ scale_v   = hvx_vec_inverse_f32(denom_v);

    #ਹਦਾਇਤ unroll(4)
    ਲਈ (ਪੂਰਨ i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼_Pair p = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(v_src[i]);
        HVX_ਸਦਿਸ਼ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_ਸਦਿਸ਼ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        v_dst[i] = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(r0, r1);
    }

    ਜੇ (nloe > 0) {
        HVX_ਸਦਿਸ਼_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_ਸਦਿਸ਼ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ਸਦਿਸ਼_Pair p = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(v1);
        HVX_ਸਦਿਸ਼ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_ਸਦਿਸ਼ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        HVX_ਸਦਿਸ਼ ਨਤੀਜਾ = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, ਨਤੀਜਾ);
    }
}

#ਜੇ_ਅੰਤ // HVX_NORM_H
