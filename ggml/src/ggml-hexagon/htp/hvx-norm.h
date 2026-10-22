#اگر_تعریف_نشده HVX_NORM_H
#تعریف_کن HVX_NORM_H

#ضمن <stdint.h>
#ضمن "hvx-base.h"
#ضمن "hvx-reduce.h"
#ضمن "hvx-inverse.h"
#ضمن "hvx-sqrt.h"
#ضمن "hvx-repl.h"

ساكن مضمن فراغ hvx_fast_rms_norm_f32(ثابت uint8_t * محدود src,
                                         uint8_t * محدود dst,
                                         ثابت صحيح رقم_elems,
                                         عائم     epsilon) {

    ثابت HVX_وکتور * محدود v_src = (HVX_وکتور *) src;
    HVX_وکتور * محدود v_dst       = (HVX_وکتور *) dst;

    ثابت صحيح nvec = رقم_elems / VLEN_FP32;    // number of full vectors
    ثابت صحيح nloe = رقم_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares for full vectors
    HVX_وکتور جمع_v = Q6_V_vsplat_R(0x00000000);
    HVX_وکتور epsilon_v = hvx_vec_splat_f32(epsilon);

    #رهنمود unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_وکتور v1 = v_src[i];
        HVX_وکتور v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        جمع_v = Q6_Vqf32_vadd_Vqf32Vqf32(جمع_v, v2);
    }

    // Handle tail elements using vectorized ops with masking
    إذا (nloe > 0) {
        HVX_وکتور_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_وکتور v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_وکتور v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        جمع_v = Q6_Vqf32_vadd_Vqf32Vqf32(جمع_v, v2);
    }

    // Reduce HVX sum
    جمع_v = hvx_vec_reduce_جمع_f32(Q6_Vsf_equals_Vqf32(جمع_v));

    HVX_وکتور t_v            = hvx_vec_splat_f32((عائم) رقم_elems);
    HVX_وکتور denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_وکتور میانگین_v         = Q6_Vqf32_vmpy_VsfVsf(جمع_v, denom_v);
    HVX_وکتور میانگین_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(میانگین_v, epsilon_v);

    // Scale full vectors
    HVX_وکتور scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(میانگین_epsilon_v));

    #رهنمود unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_وکتور v1 = v_src[i];
        HVX_وکتور v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        v_dst[i] = Q6_Vsf_equals_Vqf32(v2);
    }

    // Handle tail elements using vectorized ops with masking
    إذا (nloe > 0) {
        HVX_وکتور_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_وکتور v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_وکتور v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_وکتور نتیجه = Q6_Vsf_equals_Vqf32(v2);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, نتیجه);
    }
}

ساكن مضمن فراغ hvx_fast_rms_norm_mul_f32(ثابت uint8_t * محدود src,
                                             ثابت uint8_t * محدود وزن,
                                             uint8_t * محدود dst,
                                             ثابت صحيح رقم_elems,
                                             عائم     epsilon) {
    ثابت HVX_وکتور * محدود v_src    = (ثابت HVX_وکتور *) src;
    ثابت HVX_وکتور * محدود v_وزن = (ثابت HVX_وکتور *) وزن;
    HVX_وکتور * محدود v_dst          = (HVX_وکتور *) dst;

    ثابت صحيح nvec = رقم_elems / VLEN_FP32;    // number of full vectors
    ثابت صحيح nloe = رقم_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares for full vectors
    HVX_وکتور جمع_v = Q6_V_vsplat_R(0x00000000);
    HVX_وکتور epsilon_v = hvx_vec_splat_f32(epsilon);

    #رهنمود unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_وکتور v1 = v_src[i];
        HVX_وکتور v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        جمع_v = Q6_Vqf32_vadd_Vqf32Vqf32(جمع_v, v2);
    }

    // Handle tail elements using vectorized ops with masking
    إذا (nloe > 0) {
        HVX_وکتور_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_وکتور v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_وکتور v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        جمع_v = Q6_Vqf32_vadd_Vqf32Vqf32(جمع_v, v2);
    }

    // Reduce HVX sum
    جمع_v = hvx_vec_reduce_جمع_f32(Q6_Vsf_equals_Vqf32(جمع_v));

    HVX_وکتور t_v            = hvx_vec_splat_f32((عائم) رقم_elems);
    HVX_وکتور denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_وکتور میانگین_v         = Q6_Vqf32_vmpy_VsfVsf(جمع_v, denom_v);
    HVX_وکتور میانگین_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(میانگین_v, epsilon_v);

    // Scale and multiply
    HVX_وکتور scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(میانگین_epsilon_v));

    #رهنمود unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_وکتور v1 = v_src[i];
        HVX_وکتور v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_وکتور v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_وکتور نتیجه = Q6_Vqf32_vmpy_VsfVsf(v3, v_وزن[i]);
        v_dst[i] = Q6_Vsf_equals_Vqf32(نتیجه);
    }

    // Handle tail elements using vectorized ops with masking
    إذا (nloe > 0) {
        HVX_وکتور_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_وکتور v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_وکتور v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_وکتور v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_وکتور نتیجه = Q6_Vqf32_vmpy_VsfVsf(v3, v_وزن[nvec]);
        HVX_وکتور res_v = Q6_Vsf_equals_Vqf32(نتیجه);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, res_v);
    }
}

ساكن مضمن فراغ hvx_fast_norm_f32(ثابت uint8_t * محدود src,
                                     uint8_t * محدود dst,
                                     ثابت صحيح رقم_elems,
                                     عائم     epsilon) {

    ثابت HVX_وکتور * محدود v_src = (HVX_وکتور *) src;
    HVX_وکتور * محدود v_dst       = (HVX_وکتور *) dst;

    ثابت صحيح nvec = رقم_elems / VLEN_FP32;    // number of full vectors
    ثابت صحيح nloe = رقم_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares and sum of values for full vectors
    HVX_وکتور جمع_sq_v = Q6_V_vsplat_R(0x00000000);
    HVX_وکتور جمع_x_v  = Q6_V_vsplat_R(0x00000000);
    HVX_وکتور epsilon_v = hvx_vec_splat_f32(epsilon);

    #رهنمود unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_وکتور v1 = v_src[i];
        HVX_وکتور v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        جمع_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(جمع_sq_v, v2);
        جمع_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(جمع_x_v,  Q6_Vqf32_vadd_VsfVsf(v1, Q6_V_vzero()));
    }

    // Handle tail elements using vectorized ops with masking
    إذا (nloe > 0) {
        HVX_وکتور_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_وکتور v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_وکتور v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        جمع_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(جمع_sq_v, v2);
        جمع_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(جمع_x_v,  Q6_Vqf32_vadd_VsfVsf(v1, Q6_V_vzero()));
    }

    // Reduce HVX sums
    جمع_sq_v = hvx_vec_reduce_جمع_f32(Q6_Vsf_equals_Vqf32(جمع_sq_v));
    جمع_x_v  = hvx_vec_reduce_جمع_f32(Q6_Vsf_equals_Vqf32(جمع_x_v));

    HVX_وکتور t_v            = hvx_vec_splat_f32((عائم) رقم_elems);
    HVX_وکتور denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_وکتور میانگین_sq_v      = Q6_Vqf32_vmpy_VsfVsf(جمع_sq_v, denom_v);
    HVX_وکتور میانگین_x_v       = Q6_Vqf32_vmpy_VsfVsf(جمع_x_v,  denom_v);
    HVX_وکتور میانگین_x_sq_v    = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(میانگین_x_v), Q6_Vsf_equals_Vqf32(میانگین_x_v));
    HVX_وکتور var_v          = Q6_Vqf32_vsub_Vqf32Vqf32(میانگین_sq_v, میانگین_x_sq_v);
    HVX_وکتور var_epsilon_v  = Q6_Vqf32_vadd_Vqf32Vsf(var_v, epsilon_v);

    // scale = rsqrt(variance + epsilon),  mean_x broadcast for subtraction
    HVX_وکتور scale_v  = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(var_epsilon_v));
    HVX_وکتور میانگین_x_b = hvx_vec_repl_f32(Q6_Vsf_equals_Vqf32(میانگین_x_v));

    #رهنمود unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_وکتور v1 = v_src[i];
        HVX_وکتور v2 = Q6_Vqf32_vsub_VsfVsf(v1, میانگین_x_b);
        HVX_وکتور v3 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v2), scale_v);
        v_dst[i] = Q6_Vsf_equals_Vqf32(v3);
    }

    // Handle tail elements using vectorized ops with masking
    إذا (nloe > 0) {
        HVX_وکتور_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_وکتور v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_وکتور v2 = Q6_Vqf32_vsub_VsfVsf(v1, میانگین_x_b);
        HVX_وکتور v3 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v2), scale_v);
        HVX_وکتور نتیجه = Q6_Vsf_equals_Vqf32(v3);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, نتیجه);
    }
}

ساكن مضمن فراغ hvx_fast_l2_norm_f32(ثابت uint8_t * محدود src,
                                        uint8_t * محدود dst,
                                        ثابت صحيح رقم_elems,
                                        عائم     epsilon) {

    ثابت HVX_وکتور * محدود v_src = (HVX_وکتور *) src;
    HVX_وکتور * محدود v_dst       = (HVX_وکتور *) dst;

    HVX_وکتور جمع_v = hvx_vec_splat_f32(0.0f);

    ثابت صحيح nvec = رقم_elems / VLEN_FP32;
    ثابت صحيح nloe = رقم_elems % VLEN_FP32;

    #رهنمود unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_وکتور v1 = v_src[i];
        HVX_وکتور sq = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        جمع_v         = Q6_Vqf32_vadd_Vqf32Vqf32(جمع_v, sq);
    }

    // Include tail elements in the sum-of-squares using a predicate mask
    إذا (nloe > 0) {
        HVX_وکتور_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_وکتور v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_وکتور sq = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        جمع_v         = Q6_Vqf32_vadd_Vqf32Vqf32(جمع_v, sq);
    }

    // Compute scale = 1/fmax(sqrt(sum), epsilon) entirely in HVX registers.
    // hvx_vec_rsqrt_f32 + hvx_vec_inverse_f32 avoids scalar extraction.
    HVX_وکتور جمع_sf    = hvx_vec_reduce_جمع_f32(Q6_Vsf_equals_Vqf32(جمع_v));
    HVX_وکتور rsqrt_v   = hvx_vec_rsqrt_f32(جمع_sf);              // 1/sqrt(sum)
    HVX_وکتور sqrt_v    = hvx_vec_inverse_f32(rsqrt_v);            // sqrt(sum)
    HVX_وکتور epsilon_v = hvx_vec_splat_f32(epsilon);
    HVX_وکتور denom_v   = Q6_Vsf_vmax_VsfVsf(sqrt_v, epsilon_v);  // fmax(sqrt(sum), epsilon)
    HVX_وکتور scale_v   = hvx_vec_inverse_f32(denom_v);            // 1/fmax(sqrt(sum), epsilon)

    #رهنمود unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_وکتور v1 = v_src[i];
        v_dst[i]      = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, scale_v));
    }

    إذا (nloe > 0) {
        HVX_وکتور_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_وکتور v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_وکتور نتیجه = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, scale_v));
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, نتیجه);
    }
}

// F16 norm kernels: reduce and scale in f32 (via promote/narrow), matching the
// precision-preserving pattern used by the flash-attn f16 kernels.

ساكن مضمن فراغ hvx_fast_rms_norm_f16(ثابت uint8_t * محدود src,
                                         uint8_t * محدود dst,
                                         ثابت صحيح رقم_elems,
                                         عائم     epsilon) {

    ثابت HVX_وکتور * محدود v_src = (HVX_وکتور *) src;
    HVX_وکتور * محدود v_dst       = (HVX_وکتور *) dst;

    ثابت صحيح nvec = رقم_elems / VLEN_FP16;    // number of full f16 vectors
    ثابت صحيح nloe = رقم_elems % VLEN_FP16;    // leftover elements

    HVX_وکتور جمع_v = Q6_V_vsplat_R(0x00000000);
    HVX_وکتور epsilon_v = hvx_vec_splat_f32(epsilon);

    #رهنمود unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_وکتور_Pair p = hvx_vec_f16_مقصد_f32(v_src[i]);
        HVX_وکتور p0 = Q6_V_lo_W(p);
        HVX_وکتور p1 = Q6_V_hi_W(p);
        جمع_v = Q6_Vqf32_vadd_Vqf32Vqf32(جمع_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        جمع_v = Q6_Vqf32_vadd_Vqf32Vqf32(جمع_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    إذا (nloe > 0) {
        HVX_وکتور_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_وکتور v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_وکتور_Pair p = hvx_vec_f16_مقصد_f32(v1);
        HVX_وکتور p0 = Q6_V_lo_W(p);
        HVX_وکتور p1 = Q6_V_hi_W(p);
        جمع_v = Q6_Vqf32_vadd_Vqf32Vqf32(جمع_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        جمع_v = Q6_Vqf32_vadd_Vqf32Vqf32(جمع_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    جمع_v = hvx_vec_reduce_جمع_f32(Q6_Vsf_equals_Vqf32(جمع_v));

    HVX_وکتور t_v            = hvx_vec_splat_f32((عائم) رقم_elems);
    HVX_وکتور denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_وکتور میانگین_v         = Q6_Vqf32_vmpy_VsfVsf(جمع_v, denom_v);
    HVX_وکتور میانگین_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(میانگین_v, epsilon_v);

    HVX_وکتور scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(میانگین_epsilon_v));

    #رهنمود unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_وکتور_Pair p = hvx_vec_f16_مقصد_f32(v_src[i]);
        HVX_وکتور r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_وکتور r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        v_dst[i] = hvx_vec_f32_مقصد_f16(r0, r1);
    }

    إذا (nloe > 0) {
        HVX_وکتور_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_وکتور v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_وکتور_Pair p = hvx_vec_f16_مقصد_f32(v1);
        HVX_وکتور r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_وکتور r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        HVX_وکتور نتیجه = hvx_vec_f32_مقصد_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, نتیجه);
    }
}

ساكن مضمن فراغ hvx_fast_norm_f16(ثابت uint8_t * محدود src,
                                     uint8_t * محدود dst,
                                     ثابت صحيح رقم_elems,
                                     عائم     epsilon) {

    ثابت HVX_وکتور * محدود v_src = (HVX_وکتور *) src;
    HVX_وکتور * محدود v_dst       = (HVX_وکتور *) dst;

    ثابت صحيح nvec = رقم_elems / VLEN_FP16;
    ثابت صحيح nloe = رقم_elems % VLEN_FP16;

    HVX_وکتور جمع_sq_v = Q6_V_vsplat_R(0x00000000);
    HVX_وکتور جمع_x_v  = Q6_V_vsplat_R(0x00000000);
    HVX_وکتور epsilon_v = hvx_vec_splat_f32(epsilon);

    #رهنمود unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_وکتور_Pair p = hvx_vec_f16_مقصد_f32(v_src[i]);
        HVX_وکتور p0 = Q6_V_lo_W(p);
        HVX_وکتور p1 = Q6_V_hi_W(p);
        جمع_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(جمع_sq_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        جمع_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(جمع_sq_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
        جمع_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(جمع_x_v,  Q6_Vqf32_vadd_VsfVsf(p0, Q6_V_vzero()));
        جمع_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(جمع_x_v,  Q6_Vqf32_vadd_VsfVsf(p1, Q6_V_vzero()));
    }

    إذا (nloe > 0) {
        HVX_وکتور_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_وکتور v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_وکتور_Pair p = hvx_vec_f16_مقصد_f32(v1);
        HVX_وکتور p0 = Q6_V_lo_W(p);
        HVX_وکتور p1 = Q6_V_hi_W(p);
        جمع_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(جمع_sq_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        جمع_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(جمع_sq_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
        جمع_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(جمع_x_v,  Q6_Vqf32_vadd_VsfVsf(p0, Q6_V_vzero()));
        جمع_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(جمع_x_v,  Q6_Vqf32_vadd_VsfVsf(p1, Q6_V_vzero()));
    }

    جمع_sq_v = hvx_vec_reduce_جمع_f32(Q6_Vsf_equals_Vqf32(جمع_sq_v));
    جمع_x_v  = hvx_vec_reduce_جمع_f32(Q6_Vsf_equals_Vqf32(جمع_x_v));

    HVX_وکتور t_v            = hvx_vec_splat_f32((عائم) رقم_elems);
    HVX_وکتور denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_وکتور میانگین_sq_v      = Q6_Vqf32_vmpy_VsfVsf(جمع_sq_v, denom_v);
    HVX_وکتور میانگین_x_v       = Q6_Vqf32_vmpy_VsfVsf(جمع_x_v,  denom_v);
    HVX_وکتور میانگین_x_sq_v    = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(میانگین_x_v), Q6_Vsf_equals_Vqf32(میانگین_x_v));
    HVX_وکتور var_v          = Q6_Vqf32_vsub_Vqf32Vqf32(میانگین_sq_v, میانگین_x_sq_v);
    HVX_وکتور var_epsilon_v  = Q6_Vqf32_vadd_Vqf32Vsf(var_v, epsilon_v);

    HVX_وکتور scale_v  = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(var_epsilon_v));
    HVX_وکتور میانگین_x_b = hvx_vec_repl_f32(Q6_Vsf_equals_Vqf32(میانگین_x_v));

    #رهنمود unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_وکتور_Pair p = hvx_vec_f16_مقصد_f32(v_src[i]);
        HVX_وکتور d0 = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(p), میانگین_x_b);
        HVX_وکتور d1 = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(p), میانگین_x_b);
        HVX_وکتور r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d0), scale_v));
        HVX_وکتور r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d1), scale_v));
        v_dst[i] = hvx_vec_f32_مقصد_f16(r0, r1);
    }

    إذا (nloe > 0) {
        HVX_وکتور_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_وکتور v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_وکتور_Pair p = hvx_vec_f16_مقصد_f32(v1);
        HVX_وکتور d0 = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(p), میانگین_x_b);
        HVX_وکتور d1 = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(p), میانگین_x_b);
        HVX_وکتور r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d0), scale_v));
        HVX_وکتور r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d1), scale_v));
        HVX_وکتور نتیجه = hvx_vec_f32_مقصد_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, نتیجه);
    }
}

ساكن مضمن فراغ hvx_fast_l2_norm_f16(ثابت uint8_t * محدود src,
                                        uint8_t * محدود dst,
                                        ثابت صحيح رقم_elems,
                                        عائم     epsilon) {

    ثابت HVX_وکتور * محدود v_src = (HVX_وکتور *) src;
    HVX_وکتور * محدود v_dst       = (HVX_وکتور *) dst;

    ثابت صحيح nvec = رقم_elems / VLEN_FP16;
    ثابت صحيح nloe = رقم_elems % VLEN_FP16;

    HVX_وکتور جمع_v = hvx_vec_splat_f32(0.0f);

    #رهنمود unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_وکتور_Pair p = hvx_vec_f16_مقصد_f32(v_src[i]);
        HVX_وکتور p0 = Q6_V_lo_W(p);
        HVX_وکتور p1 = Q6_V_hi_W(p);
        جمع_v = Q6_Vqf32_vadd_Vqf32Vqf32(جمع_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        جمع_v = Q6_Vqf32_vadd_Vqf32Vqf32(جمع_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    إذا (nloe > 0) {
        HVX_وکتور_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_وکتور v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_وکتور_Pair p = hvx_vec_f16_مقصد_f32(v1);
        HVX_وکتور p0 = Q6_V_lo_W(p);
        HVX_وکتور p1 = Q6_V_hi_W(p);
        جمع_v = Q6_Vqf32_vadd_Vqf32Vqf32(جمع_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        جمع_v = Q6_Vqf32_vadd_Vqf32Vqf32(جمع_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    HVX_وکتور جمع_sf    = hvx_vec_reduce_جمع_f32(Q6_Vsf_equals_Vqf32(جمع_v));
    HVX_وکتور rsqrt_v   = hvx_vec_rsqrt_f32(جمع_sf);
    HVX_وکتور sqrt_v    = hvx_vec_inverse_f32(rsqrt_v);
    HVX_وکتور epsilon_v = hvx_vec_splat_f32(epsilon);
    HVX_وکتور denom_v   = Q6_Vsf_vmax_VsfVsf(sqrt_v, epsilon_v);
    HVX_وکتور scale_v   = hvx_vec_inverse_f32(denom_v);

    #رهنمود unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_وکتور_Pair p = hvx_vec_f16_مقصد_f32(v_src[i]);
        HVX_وکتور r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_وکتور r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        v_dst[i] = hvx_vec_f32_مقصد_f16(r0, r1);
    }

    إذا (nloe > 0) {
        HVX_وکتور_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_وکتور v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_وکتور_Pair p = hvx_vec_f16_مقصد_f32(v1);
        HVX_وکتور r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_وکتور r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        HVX_وکتور نتیجه = hvx_vec_f32_مقصد_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, نتیجه);
    }
}

#پایان_اگر // HVX_NORM_H
