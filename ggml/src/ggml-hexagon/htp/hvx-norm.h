#که_نه_تعریف_شوی HVX_NORM_H
#تعریف_کړه HVX_NORM_H

#ضمن <stdint.h>
#ضمن "hvx-base.h"
#ضمن "hvx-reduce.h"
#ضمن "hvx-inverse.h"
#ضمن "hvx-sqrt.h"
#ضمن "hvx-repl.h"

ساكن مضمن فراغ hvx_fast_rms_norm_f32(ثابت uint8_t * محدود src,
                                         uint8_t * محدود dst,
                                         ثابت صحيح عدد_elems,
                                         عائم     epsilon) {

    ثابت HVX_ویکټور * محدود v_src = (HVX_ویکټور *) src;
    HVX_ویکټور * محدود v_dst       = (HVX_ویکټور *) dst;

    ثابت صحيح nvec = عدد_elems / VLEN_FP32;    // number of full vectors
    ثابت صحيح nloe = عدد_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares for full vectors
    HVX_ویکټور مجموعه_v = Q6_V_vsplat_R(0x00000000);
    HVX_ویکټور epsilon_v = hvx_vec_splat_f32(epsilon);

    #لارښوونه unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ویکټور v1 = v_src[i];
        HVX_ویکټور v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        مجموعه_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموعه_v, v2);
    }

    // Handle tail elements using vectorized ops with masking
    إذا (nloe > 0) {
        HVX_ویکټور_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ویکټور v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ویکټور v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        مجموعه_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموعه_v, v2);
    }

    // Reduce HVX sum
    مجموعه_v = hvx_vec_reduce_مجموعه_f32(Q6_Vsf_equals_Vqf32(مجموعه_v));

    HVX_ویکټور t_v            = hvx_vec_splat_f32((عائم) عدد_elems);
    HVX_ویکټور denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_ویکټور منځنۍ_v         = Q6_Vqf32_vmpy_VsfVsf(مجموعه_v, denom_v);
    HVX_ویکټور منځنۍ_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(منځنۍ_v, epsilon_v);

    // Scale full vectors
    HVX_ویکټور scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(منځنۍ_epsilon_v));

    #لارښوونه unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ویکټور v1 = v_src[i];
        HVX_ویکټور v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        v_dst[i] = Q6_Vsf_equals_Vqf32(v2);
    }

    // Handle tail elements using vectorized ops with masking
    إذا (nloe > 0) {
        HVX_ویکټور_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ویکټور v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ویکټور v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_ویکټور پایله = Q6_Vsf_equals_Vqf32(v2);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, پایله);
    }
}

ساكن مضمن فراغ hvx_fast_rms_norm_mul_f32(ثابت uint8_t * محدود src,
                                             ثابت uint8_t * محدود وزن,
                                             uint8_t * محدود dst,
                                             ثابت صحيح عدد_elems,
                                             عائم     epsilon) {
    ثابت HVX_ویکټور * محدود v_src    = (ثابت HVX_ویکټور *) src;
    ثابت HVX_ویکټور * محدود v_وزن = (ثابت HVX_ویکټور *) وزن;
    HVX_ویکټور * محدود v_dst          = (HVX_ویکټور *) dst;

    ثابت صحيح nvec = عدد_elems / VLEN_FP32;    // number of full vectors
    ثابت صحيح nloe = عدد_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares for full vectors
    HVX_ویکټور مجموعه_v = Q6_V_vsplat_R(0x00000000);
    HVX_ویکټور epsilon_v = hvx_vec_splat_f32(epsilon);

    #لارښوونه unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ویکټور v1 = v_src[i];
        HVX_ویکټور v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        مجموعه_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموعه_v, v2);
    }

    // Handle tail elements using vectorized ops with masking
    إذا (nloe > 0) {
        HVX_ویکټور_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ویکټور v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ویکټور v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        مجموعه_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموعه_v, v2);
    }

    // Reduce HVX sum
    مجموعه_v = hvx_vec_reduce_مجموعه_f32(Q6_Vsf_equals_Vqf32(مجموعه_v));

    HVX_ویکټور t_v            = hvx_vec_splat_f32((عائم) عدد_elems);
    HVX_ویکټور denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_ویکټور منځنۍ_v         = Q6_Vqf32_vmpy_VsfVsf(مجموعه_v, denom_v);
    HVX_ویکټور منځنۍ_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(منځنۍ_v, epsilon_v);

    // Scale and multiply
    HVX_ویکټور scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(منځنۍ_epsilon_v));

    #لارښوونه unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ویکټور v1 = v_src[i];
        HVX_ویکټور v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_ویکټور v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_ویکټور پایله = Q6_Vqf32_vmpy_VsfVsf(v3, v_وزن[i]);
        v_dst[i] = Q6_Vsf_equals_Vqf32(پایله);
    }

    // Handle tail elements using vectorized ops with masking
    إذا (nloe > 0) {
        HVX_ویکټور_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ویکټور v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ویکټور v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_ویکټور v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_ویکټور پایله = Q6_Vqf32_vmpy_VsfVsf(v3, v_وزن[nvec]);
        HVX_ویکټور res_v = Q6_Vsf_equals_Vqf32(پایله);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, res_v);
    }
}

ساكن مضمن فراغ hvx_fast_norm_f32(ثابت uint8_t * محدود src,
                                     uint8_t * محدود dst,
                                     ثابت صحيح عدد_elems,
                                     عائم     epsilon) {

    ثابت HVX_ویکټور * محدود v_src = (HVX_ویکټور *) src;
    HVX_ویکټور * محدود v_dst       = (HVX_ویکټور *) dst;

    ثابت صحيح nvec = عدد_elems / VLEN_FP32;    // number of full vectors
    ثابت صحيح nloe = عدد_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares and sum of values for full vectors
    HVX_ویکټور مجموعه_sq_v = Q6_V_vsplat_R(0x00000000);
    HVX_ویکټور مجموعه_x_v  = Q6_V_vsplat_R(0x00000000);
    HVX_ویکټور epsilon_v = hvx_vec_splat_f32(epsilon);

    #لارښوونه unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ویکټور v1 = v_src[i];
        HVX_ویکټور v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        مجموعه_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموعه_sq_v, v2);
        مجموعه_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(مجموعه_x_v,  Q6_Vqf32_vadd_VsfVsf(v1, Q6_V_vzero()));
    }

    // Handle tail elements using vectorized ops with masking
    إذا (nloe > 0) {
        HVX_ویکټور_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ویکټور v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ویکټور v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        مجموعه_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموعه_sq_v, v2);
        مجموعه_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(مجموعه_x_v,  Q6_Vqf32_vadd_VsfVsf(v1, Q6_V_vzero()));
    }

    // Reduce HVX sums
    مجموعه_sq_v = hvx_vec_reduce_مجموعه_f32(Q6_Vsf_equals_Vqf32(مجموعه_sq_v));
    مجموعه_x_v  = hvx_vec_reduce_مجموعه_f32(Q6_Vsf_equals_Vqf32(مجموعه_x_v));

    HVX_ویکټور t_v            = hvx_vec_splat_f32((عائم) عدد_elems);
    HVX_ویکټور denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_ویکټور منځنۍ_sq_v      = Q6_Vqf32_vmpy_VsfVsf(مجموعه_sq_v, denom_v);
    HVX_ویکټور منځنۍ_x_v       = Q6_Vqf32_vmpy_VsfVsf(مجموعه_x_v,  denom_v);
    HVX_ویکټور منځنۍ_x_sq_v    = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(منځنۍ_x_v), Q6_Vsf_equals_Vqf32(منځنۍ_x_v));
    HVX_ویکټور var_v          = Q6_Vqf32_vsub_Vqf32Vqf32(منځنۍ_sq_v, منځنۍ_x_sq_v);
    HVX_ویکټور var_epsilon_v  = Q6_Vqf32_vadd_Vqf32Vsf(var_v, epsilon_v);

    // scale = rsqrt(variance + epsilon),  mean_x broadcast for subtraction
    HVX_ویکټور scale_v  = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(var_epsilon_v));
    HVX_ویکټور منځنۍ_x_b = hvx_vec_repl_f32(Q6_Vsf_equals_Vqf32(منځنۍ_x_v));

    #لارښوونه unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ویکټور v1 = v_src[i];
        HVX_ویکټور v2 = Q6_Vqf32_vsub_VsfVsf(v1, منځنۍ_x_b);
        HVX_ویکټور v3 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v2), scale_v);
        v_dst[i] = Q6_Vsf_equals_Vqf32(v3);
    }

    // Handle tail elements using vectorized ops with masking
    إذا (nloe > 0) {
        HVX_ویکټور_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ویکټور v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ویکټور v2 = Q6_Vqf32_vsub_VsfVsf(v1, منځنۍ_x_b);
        HVX_ویکټور v3 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v2), scale_v);
        HVX_ویکټور پایله = Q6_Vsf_equals_Vqf32(v3);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, پایله);
    }
}

ساكن مضمن فراغ hvx_fast_l2_norm_f32(ثابت uint8_t * محدود src,
                                        uint8_t * محدود dst,
                                        ثابت صحيح عدد_elems,
                                        عائم     epsilon) {

    ثابت HVX_ویکټور * محدود v_src = (HVX_ویکټور *) src;
    HVX_ویکټور * محدود v_dst       = (HVX_ویکټور *) dst;

    HVX_ویکټور مجموعه_v = hvx_vec_splat_f32(0.0f);

    ثابت صحيح nvec = عدد_elems / VLEN_FP32;
    ثابت صحيح nloe = عدد_elems % VLEN_FP32;

    #لارښوونه unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ویکټور v1 = v_src[i];
        HVX_ویکټور sq = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        مجموعه_v         = Q6_Vqf32_vadd_Vqf32Vqf32(مجموعه_v, sq);
    }

    // Include tail elements in the sum-of-squares using a predicate mask
    إذا (nloe > 0) {
        HVX_ویکټور_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ویکټور v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ویکټور sq = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        مجموعه_v         = Q6_Vqf32_vadd_Vqf32Vqf32(مجموعه_v, sq);
    }

    // Compute scale = 1/fmax(sqrt(sum), epsilon) entirely in HVX registers.
    // hvx_vec_rsqrt_f32 + hvx_vec_inverse_f32 avoids scalar extraction.
    HVX_ویکټور مجموعه_sf    = hvx_vec_reduce_مجموعه_f32(Q6_Vsf_equals_Vqf32(مجموعه_v));
    HVX_ویکټور rsqrt_v   = hvx_vec_rsqrt_f32(مجموعه_sf);              // 1/sqrt(sum)
    HVX_ویکټور sqrt_v    = hvx_vec_inverse_f32(rsqrt_v);            // sqrt(sum)
    HVX_ویکټور epsilon_v = hvx_vec_splat_f32(epsilon);
    HVX_ویکټور denom_v   = Q6_Vsf_vmax_VsfVsf(sqrt_v, epsilon_v);  // fmax(sqrt(sum), epsilon)
    HVX_ویکټور scale_v   = hvx_vec_inverse_f32(denom_v);            // 1/fmax(sqrt(sum), epsilon)

    #لارښوونه unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ویکټور v1 = v_src[i];
        v_dst[i]      = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, scale_v));
    }

    إذا (nloe > 0) {
        HVX_ویکټور_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ویکټور v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ویکټور پایله = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, scale_v));
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, پایله);
    }
}

// F16 norm kernels: reduce and scale in f32 (via promote/narrow), matching the
// precision-preserving pattern used by the flash-attn f16 kernels.

ساكن مضمن فراغ hvx_fast_rms_norm_f16(ثابت uint8_t * محدود src,
                                         uint8_t * محدود dst,
                                         ثابت صحيح عدد_elems,
                                         عائم     epsilon) {

    ثابت HVX_ویکټور * محدود v_src = (HVX_ویکټور *) src;
    HVX_ویکټور * محدود v_dst       = (HVX_ویکټور *) dst;

    ثابت صحيح nvec = عدد_elems / VLEN_FP16;    // number of full f16 vectors
    ثابت صحيح nloe = عدد_elems % VLEN_FP16;    // leftover elements

    HVX_ویکټور مجموعه_v = Q6_V_vsplat_R(0x00000000);
    HVX_ویکټور epsilon_v = hvx_vec_splat_f32(epsilon);

    #لارښوونه unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ویکټور_Pair p = hvx_vec_f16_منزل_f32(v_src[i]);
        HVX_ویکټور p0 = Q6_V_lo_W(p);
        HVX_ویکټور p1 = Q6_V_hi_W(p);
        مجموعه_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموعه_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        مجموعه_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموعه_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    إذا (nloe > 0) {
        HVX_ویکټور_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_ویکټور v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ویکټور_Pair p = hvx_vec_f16_منزل_f32(v1);
        HVX_ویکټور p0 = Q6_V_lo_W(p);
        HVX_ویکټور p1 = Q6_V_hi_W(p);
        مجموعه_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموعه_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        مجموعه_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموعه_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    مجموعه_v = hvx_vec_reduce_مجموعه_f32(Q6_Vsf_equals_Vqf32(مجموعه_v));

    HVX_ویکټور t_v            = hvx_vec_splat_f32((عائم) عدد_elems);
    HVX_ویکټور denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_ویکټور منځنۍ_v         = Q6_Vqf32_vmpy_VsfVsf(مجموعه_v, denom_v);
    HVX_ویکټور منځنۍ_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(منځنۍ_v, epsilon_v);

    HVX_ویکټور scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(منځنۍ_epsilon_v));

    #لارښوونه unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ویکټور_Pair p = hvx_vec_f16_منزل_f32(v_src[i]);
        HVX_ویکټور r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_ویکټور r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        v_dst[i] = hvx_vec_f32_منزل_f16(r0, r1);
    }

    إذا (nloe > 0) {
        HVX_ویکټور_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_ویکټور v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ویکټور_Pair p = hvx_vec_f16_منزل_f32(v1);
        HVX_ویکټور r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_ویکټور r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        HVX_ویکټور پایله = hvx_vec_f32_منزل_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, پایله);
    }
}

ساكن مضمن فراغ hvx_fast_norm_f16(ثابت uint8_t * محدود src,
                                     uint8_t * محدود dst,
                                     ثابت صحيح عدد_elems,
                                     عائم     epsilon) {

    ثابت HVX_ویکټور * محدود v_src = (HVX_ویکټور *) src;
    HVX_ویکټور * محدود v_dst       = (HVX_ویکټور *) dst;

    ثابت صحيح nvec = عدد_elems / VLEN_FP16;
    ثابت صحيح nloe = عدد_elems % VLEN_FP16;

    HVX_ویکټور مجموعه_sq_v = Q6_V_vsplat_R(0x00000000);
    HVX_ویکټور مجموعه_x_v  = Q6_V_vsplat_R(0x00000000);
    HVX_ویکټور epsilon_v = hvx_vec_splat_f32(epsilon);

    #لارښوونه unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ویکټور_Pair p = hvx_vec_f16_منزل_f32(v_src[i]);
        HVX_ویکټور p0 = Q6_V_lo_W(p);
        HVX_ویکټور p1 = Q6_V_hi_W(p);
        مجموعه_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموعه_sq_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        مجموعه_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموعه_sq_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
        مجموعه_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(مجموعه_x_v,  Q6_Vqf32_vadd_VsfVsf(p0, Q6_V_vzero()));
        مجموعه_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(مجموعه_x_v,  Q6_Vqf32_vadd_VsfVsf(p1, Q6_V_vzero()));
    }

    إذا (nloe > 0) {
        HVX_ویکټور_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_ویکټور v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ویکټور_Pair p = hvx_vec_f16_منزل_f32(v1);
        HVX_ویکټور p0 = Q6_V_lo_W(p);
        HVX_ویکټور p1 = Q6_V_hi_W(p);
        مجموعه_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموعه_sq_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        مجموعه_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموعه_sq_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
        مجموعه_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(مجموعه_x_v,  Q6_Vqf32_vadd_VsfVsf(p0, Q6_V_vzero()));
        مجموعه_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(مجموعه_x_v,  Q6_Vqf32_vadd_VsfVsf(p1, Q6_V_vzero()));
    }

    مجموعه_sq_v = hvx_vec_reduce_مجموعه_f32(Q6_Vsf_equals_Vqf32(مجموعه_sq_v));
    مجموعه_x_v  = hvx_vec_reduce_مجموعه_f32(Q6_Vsf_equals_Vqf32(مجموعه_x_v));

    HVX_ویکټور t_v            = hvx_vec_splat_f32((عائم) عدد_elems);
    HVX_ویکټور denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_ویکټور منځنۍ_sq_v      = Q6_Vqf32_vmpy_VsfVsf(مجموعه_sq_v, denom_v);
    HVX_ویکټور منځنۍ_x_v       = Q6_Vqf32_vmpy_VsfVsf(مجموعه_x_v,  denom_v);
    HVX_ویکټور منځنۍ_x_sq_v    = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(منځنۍ_x_v), Q6_Vsf_equals_Vqf32(منځنۍ_x_v));
    HVX_ویکټور var_v          = Q6_Vqf32_vsub_Vqf32Vqf32(منځنۍ_sq_v, منځنۍ_x_sq_v);
    HVX_ویکټور var_epsilon_v  = Q6_Vqf32_vadd_Vqf32Vsf(var_v, epsilon_v);

    HVX_ویکټور scale_v  = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(var_epsilon_v));
    HVX_ویکټور منځنۍ_x_b = hvx_vec_repl_f32(Q6_Vsf_equals_Vqf32(منځنۍ_x_v));

    #لارښوونه unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ویکټور_Pair p = hvx_vec_f16_منزل_f32(v_src[i]);
        HVX_ویکټور d0 = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(p), منځنۍ_x_b);
        HVX_ویکټور d1 = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(p), منځنۍ_x_b);
        HVX_ویکټور r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d0), scale_v));
        HVX_ویکټور r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d1), scale_v));
        v_dst[i] = hvx_vec_f32_منزل_f16(r0, r1);
    }

    إذا (nloe > 0) {
        HVX_ویکټور_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_ویکټور v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ویکټور_Pair p = hvx_vec_f16_منزل_f32(v1);
        HVX_ویکټور d0 = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(p), منځنۍ_x_b);
        HVX_ویکټور d1 = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(p), منځنۍ_x_b);
        HVX_ویکټور r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d0), scale_v));
        HVX_ویکټور r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d1), scale_v));
        HVX_ویکټور پایله = hvx_vec_f32_منزل_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, پایله);
    }
}

ساكن مضمن فراغ hvx_fast_l2_norm_f16(ثابت uint8_t * محدود src,
                                        uint8_t * محدود dst,
                                        ثابت صحيح عدد_elems,
                                        عائم     epsilon) {

    ثابت HVX_ویکټور * محدود v_src = (HVX_ویکټور *) src;
    HVX_ویکټور * محدود v_dst       = (HVX_ویکټور *) dst;

    ثابت صحيح nvec = عدد_elems / VLEN_FP16;
    ثابت صحيح nloe = عدد_elems % VLEN_FP16;

    HVX_ویکټور مجموعه_v = hvx_vec_splat_f32(0.0f);

    #لارښوونه unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ویکټور_Pair p = hvx_vec_f16_منزل_f32(v_src[i]);
        HVX_ویکټور p0 = Q6_V_lo_W(p);
        HVX_ویکټور p1 = Q6_V_hi_W(p);
        مجموعه_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموعه_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        مجموعه_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموعه_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    إذا (nloe > 0) {
        HVX_ویکټور_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_ویکټور v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ویکټور_Pair p = hvx_vec_f16_منزل_f32(v1);
        HVX_ویکټور p0 = Q6_V_lo_W(p);
        HVX_ویکټور p1 = Q6_V_hi_W(p);
        مجموعه_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموعه_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        مجموعه_v = Q6_Vqf32_vadd_Vqf32Vqf32(مجموعه_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    HVX_ویکټور مجموعه_sf    = hvx_vec_reduce_مجموعه_f32(Q6_Vsf_equals_Vqf32(مجموعه_v));
    HVX_ویکټور rsqrt_v   = hvx_vec_rsqrt_f32(مجموعه_sf);
    HVX_ویکټور sqrt_v    = hvx_vec_inverse_f32(rsqrt_v);
    HVX_ویکټور epsilon_v = hvx_vec_splat_f32(epsilon);
    HVX_ویکټور denom_v   = Q6_Vsf_vmax_VsfVsf(sqrt_v, epsilon_v);
    HVX_ویکټور scale_v   = hvx_vec_inverse_f32(denom_v);

    #لارښوونه unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ویکټور_Pair p = hvx_vec_f16_منزل_f32(v_src[i]);
        HVX_ویکټور r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_ویکټور r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        v_dst[i] = hvx_vec_f32_منزل_f16(r0, r1);
    }

    إذا (nloe > 0) {
        HVX_ویکټور_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_ویکټور v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ویکټور_Pair p = hvx_vec_f16_منزل_f32(v1);
        HVX_ویکټور r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_ویکټور r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        HVX_ویکټور پایله = hvx_vec_f32_منزل_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, پایله);
    }
}

#د_که_پای // HVX_NORM_H
