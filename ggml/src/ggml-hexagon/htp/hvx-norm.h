#جيڪڏهن_وصف_نه_ٿيل HVX_NORM_H
#وصف_ڪريو HVX_NORM_H

#ضمن <stdint.h>
#ضمن "hvx-base.h"
#ضمن "hvx-reduce.h"
#ضمن "hvx-inverse.h"
#ضمن "hvx-sqrt.h"
#ضمن "hvx-repl.h"

ساكن مضمن فراغ hvx_fast_rms_norm_f32(ثابت uint8_t * محدود src,
                                         uint8_t * محدود dst,
                                         ثابت صحيح انگ_elems,
                                         عائم     epsilon) {

    ثابت HVX_ويڪٽر * محدود v_src = (HVX_ويڪٽر *) src;
    HVX_ويڪٽر * محدود v_dst       = (HVX_ويڪٽر *) dst;

    ثابت صحيح nvec = انگ_elems / VLEN_FP32;    // number of full vectors
    ثابت صحيح nloe = انگ_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares for full vectors
    HVX_ويڪٽر جوڙ_v = Q6_V_vsplat_R(0x00000000);
    HVX_ويڪٽر epsilon_v = hvx_vec_splat_f32(epsilon);

    #هدايت unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ويڪٽر v1 = v_src[i];
        HVX_ويڪٽر v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        جوڙ_v = Q6_Vqf32_vadd_Vqf32Vqf32(جوڙ_v, v2);
    }

    // Handle tail elements using vectorized ops with masking
    إذا (nloe > 0) {
        HVX_ويڪٽر_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ويڪٽر v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ويڪٽر v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        جوڙ_v = Q6_Vqf32_vadd_Vqf32Vqf32(جوڙ_v, v2);
    }

    // Reduce HVX sum
    جوڙ_v = hvx_vec_reduce_جوڙ_f32(Q6_Vsf_equals_Vqf32(جوڙ_v));

    HVX_ويڪٽر t_v            = hvx_vec_splat_f32((عائم) انگ_elems);
    HVX_ويڪٽر denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_ويڪٽر سراسري_v         = Q6_Vqf32_vmpy_VsfVsf(جوڙ_v, denom_v);
    HVX_ويڪٽر سراسري_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(سراسري_v, epsilon_v);

    // Scale full vectors
    HVX_ويڪٽر scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(سراسري_epsilon_v));

    #هدايت unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ويڪٽر v1 = v_src[i];
        HVX_ويڪٽر v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        v_dst[i] = Q6_Vsf_equals_Vqf32(v2);
    }

    // Handle tail elements using vectorized ops with masking
    إذا (nloe > 0) {
        HVX_ويڪٽر_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ويڪٽر v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ويڪٽر v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_ويڪٽر نتيجو = Q6_Vsf_equals_Vqf32(v2);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, نتيجو);
    }
}

ساكن مضمن فراغ hvx_fast_rms_norm_mul_f32(ثابت uint8_t * محدود src,
                                             ثابت uint8_t * محدود وزن,
                                             uint8_t * محدود dst,
                                             ثابت صحيح انگ_elems,
                                             عائم     epsilon) {
    ثابت HVX_ويڪٽر * محدود v_src    = (ثابت HVX_ويڪٽر *) src;
    ثابت HVX_ويڪٽر * محدود v_وزن = (ثابت HVX_ويڪٽر *) وزن;
    HVX_ويڪٽر * محدود v_dst          = (HVX_ويڪٽر *) dst;

    ثابت صحيح nvec = انگ_elems / VLEN_FP32;    // number of full vectors
    ثابت صحيح nloe = انگ_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares for full vectors
    HVX_ويڪٽر جوڙ_v = Q6_V_vsplat_R(0x00000000);
    HVX_ويڪٽر epsilon_v = hvx_vec_splat_f32(epsilon);

    #هدايت unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ويڪٽر v1 = v_src[i];
        HVX_ويڪٽر v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        جوڙ_v = Q6_Vqf32_vadd_Vqf32Vqf32(جوڙ_v, v2);
    }

    // Handle tail elements using vectorized ops with masking
    إذا (nloe > 0) {
        HVX_ويڪٽر_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ويڪٽر v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ويڪٽر v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        جوڙ_v = Q6_Vqf32_vadd_Vqf32Vqf32(جوڙ_v, v2);
    }

    // Reduce HVX sum
    جوڙ_v = hvx_vec_reduce_جوڙ_f32(Q6_Vsf_equals_Vqf32(جوڙ_v));

    HVX_ويڪٽر t_v            = hvx_vec_splat_f32((عائم) انگ_elems);
    HVX_ويڪٽر denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_ويڪٽر سراسري_v         = Q6_Vqf32_vmpy_VsfVsf(جوڙ_v, denom_v);
    HVX_ويڪٽر سراسري_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(سراسري_v, epsilon_v);

    // Scale and multiply
    HVX_ويڪٽر scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(سراسري_epsilon_v));

    #هدايت unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ويڪٽر v1 = v_src[i];
        HVX_ويڪٽر v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_ويڪٽر v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_ويڪٽر نتيجو = Q6_Vqf32_vmpy_VsfVsf(v3, v_وزن[i]);
        v_dst[i] = Q6_Vsf_equals_Vqf32(نتيجو);
    }

    // Handle tail elements using vectorized ops with masking
    إذا (nloe > 0) {
        HVX_ويڪٽر_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ويڪٽر v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ويڪٽر v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_ويڪٽر v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_ويڪٽر نتيجو = Q6_Vqf32_vmpy_VsfVsf(v3, v_وزن[nvec]);
        HVX_ويڪٽر res_v = Q6_Vsf_equals_Vqf32(نتيجو);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, res_v);
    }
}

ساكن مضمن فراغ hvx_fast_norm_f32(ثابت uint8_t * محدود src,
                                     uint8_t * محدود dst,
                                     ثابت صحيح انگ_elems,
                                     عائم     epsilon) {

    ثابت HVX_ويڪٽر * محدود v_src = (HVX_ويڪٽر *) src;
    HVX_ويڪٽر * محدود v_dst       = (HVX_ويڪٽر *) dst;

    ثابت صحيح nvec = انگ_elems / VLEN_FP32;    // number of full vectors
    ثابت صحيح nloe = انگ_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares and sum of values for full vectors
    HVX_ويڪٽر جوڙ_sq_v = Q6_V_vsplat_R(0x00000000);
    HVX_ويڪٽر جوڙ_x_v  = Q6_V_vsplat_R(0x00000000);
    HVX_ويڪٽر epsilon_v = hvx_vec_splat_f32(epsilon);

    #هدايت unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ويڪٽر v1 = v_src[i];
        HVX_ويڪٽر v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        جوڙ_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(جوڙ_sq_v, v2);
        جوڙ_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(جوڙ_x_v,  Q6_Vqf32_vadd_VsfVsf(v1, Q6_V_vzero()));
    }

    // Handle tail elements using vectorized ops with masking
    إذا (nloe > 0) {
        HVX_ويڪٽر_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ويڪٽر v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ويڪٽر v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        جوڙ_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(جوڙ_sq_v, v2);
        جوڙ_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(جوڙ_x_v,  Q6_Vqf32_vadd_VsfVsf(v1, Q6_V_vzero()));
    }

    // Reduce HVX sums
    جوڙ_sq_v = hvx_vec_reduce_جوڙ_f32(Q6_Vsf_equals_Vqf32(جوڙ_sq_v));
    جوڙ_x_v  = hvx_vec_reduce_جوڙ_f32(Q6_Vsf_equals_Vqf32(جوڙ_x_v));

    HVX_ويڪٽر t_v            = hvx_vec_splat_f32((عائم) انگ_elems);
    HVX_ويڪٽر denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_ويڪٽر سراسري_sq_v      = Q6_Vqf32_vmpy_VsfVsf(جوڙ_sq_v, denom_v);
    HVX_ويڪٽر سراسري_x_v       = Q6_Vqf32_vmpy_VsfVsf(جوڙ_x_v,  denom_v);
    HVX_ويڪٽر سراسري_x_sq_v    = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(سراسري_x_v), Q6_Vsf_equals_Vqf32(سراسري_x_v));
    HVX_ويڪٽر var_v          = Q6_Vqf32_vsub_Vqf32Vqf32(سراسري_sq_v, سراسري_x_sq_v);
    HVX_ويڪٽر var_epsilon_v  = Q6_Vqf32_vadd_Vqf32Vsf(var_v, epsilon_v);

    // scale = rsqrt(variance + epsilon),  mean_x broadcast for subtraction
    HVX_ويڪٽر scale_v  = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(var_epsilon_v));
    HVX_ويڪٽر سراسري_x_b = hvx_vec_repl_f32(Q6_Vsf_equals_Vqf32(سراسري_x_v));

    #هدايت unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ويڪٽر v1 = v_src[i];
        HVX_ويڪٽر v2 = Q6_Vqf32_vsub_VsfVsf(v1, سراسري_x_b);
        HVX_ويڪٽر v3 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v2), scale_v);
        v_dst[i] = Q6_Vsf_equals_Vqf32(v3);
    }

    // Handle tail elements using vectorized ops with masking
    إذا (nloe > 0) {
        HVX_ويڪٽر_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ويڪٽر v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ويڪٽر v2 = Q6_Vqf32_vsub_VsfVsf(v1, سراسري_x_b);
        HVX_ويڪٽر v3 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v2), scale_v);
        HVX_ويڪٽر نتيجو = Q6_Vsf_equals_Vqf32(v3);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, نتيجو);
    }
}

ساكن مضمن فراغ hvx_fast_l2_norm_f32(ثابت uint8_t * محدود src,
                                        uint8_t * محدود dst,
                                        ثابت صحيح انگ_elems,
                                        عائم     epsilon) {

    ثابت HVX_ويڪٽر * محدود v_src = (HVX_ويڪٽر *) src;
    HVX_ويڪٽر * محدود v_dst       = (HVX_ويڪٽر *) dst;

    HVX_ويڪٽر جوڙ_v = hvx_vec_splat_f32(0.0f);

    ثابت صحيح nvec = انگ_elems / VLEN_FP32;
    ثابت صحيح nloe = انگ_elems % VLEN_FP32;

    #هدايت unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ويڪٽر v1 = v_src[i];
        HVX_ويڪٽر sq = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        جوڙ_v         = Q6_Vqf32_vadd_Vqf32Vqf32(جوڙ_v, sq);
    }

    // Include tail elements in the sum-of-squares using a predicate mask
    إذا (nloe > 0) {
        HVX_ويڪٽر_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ويڪٽر v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ويڪٽر sq = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        جوڙ_v         = Q6_Vqf32_vadd_Vqf32Vqf32(جوڙ_v, sq);
    }

    // Compute scale = 1/fmax(sqrt(sum), epsilon) entirely in HVX registers.
    // hvx_vec_rsqrt_f32 + hvx_vec_inverse_f32 avoids scalar extraction.
    HVX_ويڪٽر جوڙ_sf    = hvx_vec_reduce_جوڙ_f32(Q6_Vsf_equals_Vqf32(جوڙ_v));
    HVX_ويڪٽر rsqrt_v   = hvx_vec_rsqrt_f32(جوڙ_sf);              // 1/sqrt(sum)
    HVX_ويڪٽر sqrt_v    = hvx_vec_inverse_f32(rsqrt_v);            // sqrt(sum)
    HVX_ويڪٽر epsilon_v = hvx_vec_splat_f32(epsilon);
    HVX_ويڪٽر denom_v   = Q6_Vsf_vmax_VsfVsf(sqrt_v, epsilon_v);  // fmax(sqrt(sum), epsilon)
    HVX_ويڪٽر scale_v   = hvx_vec_inverse_f32(denom_v);            // 1/fmax(sqrt(sum), epsilon)

    #هدايت unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ويڪٽر v1 = v_src[i];
        v_dst[i]      = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, scale_v));
    }

    إذا (nloe > 0) {
        HVX_ويڪٽر_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ويڪٽر v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ويڪٽر نتيجو = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, scale_v));
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, نتيجو);
    }
}

// F16 norm kernels: reduce and scale in f32 (via promote/narrow), matching the
// precision-preserving pattern used by the flash-attn f16 kernels.

ساكن مضمن فراغ hvx_fast_rms_norm_f16(ثابت uint8_t * محدود src,
                                         uint8_t * محدود dst,
                                         ثابت صحيح انگ_elems,
                                         عائم     epsilon) {

    ثابت HVX_ويڪٽر * محدود v_src = (HVX_ويڪٽر *) src;
    HVX_ويڪٽر * محدود v_dst       = (HVX_ويڪٽر *) dst;

    ثابت صحيح nvec = انگ_elems / VLEN_FP16;    // number of full f16 vectors
    ثابت صحيح nloe = انگ_elems % VLEN_FP16;    // leftover elements

    HVX_ويڪٽر جوڙ_v = Q6_V_vsplat_R(0x00000000);
    HVX_ويڪٽر epsilon_v = hvx_vec_splat_f32(epsilon);

    #هدايت unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ويڪٽر_Pair p = hvx_vec_f16_منزل_f32(v_src[i]);
        HVX_ويڪٽر p0 = Q6_V_lo_W(p);
        HVX_ويڪٽر p1 = Q6_V_hi_W(p);
        جوڙ_v = Q6_Vqf32_vadd_Vqf32Vqf32(جوڙ_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        جوڙ_v = Q6_Vqf32_vadd_Vqf32Vqf32(جوڙ_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    إذا (nloe > 0) {
        HVX_ويڪٽر_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_ويڪٽر v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ويڪٽر_Pair p = hvx_vec_f16_منزل_f32(v1);
        HVX_ويڪٽر p0 = Q6_V_lo_W(p);
        HVX_ويڪٽر p1 = Q6_V_hi_W(p);
        جوڙ_v = Q6_Vqf32_vadd_Vqf32Vqf32(جوڙ_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        جوڙ_v = Q6_Vqf32_vadd_Vqf32Vqf32(جوڙ_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    جوڙ_v = hvx_vec_reduce_جوڙ_f32(Q6_Vsf_equals_Vqf32(جوڙ_v));

    HVX_ويڪٽر t_v            = hvx_vec_splat_f32((عائم) انگ_elems);
    HVX_ويڪٽر denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_ويڪٽر سراسري_v         = Q6_Vqf32_vmpy_VsfVsf(جوڙ_v, denom_v);
    HVX_ويڪٽر سراسري_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(سراسري_v, epsilon_v);

    HVX_ويڪٽر scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(سراسري_epsilon_v));

    #هدايت unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ويڪٽر_Pair p = hvx_vec_f16_منزل_f32(v_src[i]);
        HVX_ويڪٽر r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_ويڪٽر r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        v_dst[i] = hvx_vec_f32_منزل_f16(r0, r1);
    }

    إذا (nloe > 0) {
        HVX_ويڪٽر_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_ويڪٽر v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ويڪٽر_Pair p = hvx_vec_f16_منزل_f32(v1);
        HVX_ويڪٽر r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_ويڪٽر r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        HVX_ويڪٽر نتيجو = hvx_vec_f32_منزل_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, نتيجو);
    }
}

ساكن مضمن فراغ hvx_fast_norm_f16(ثابت uint8_t * محدود src,
                                     uint8_t * محدود dst,
                                     ثابت صحيح انگ_elems,
                                     عائم     epsilon) {

    ثابت HVX_ويڪٽر * محدود v_src = (HVX_ويڪٽر *) src;
    HVX_ويڪٽر * محدود v_dst       = (HVX_ويڪٽر *) dst;

    ثابت صحيح nvec = انگ_elems / VLEN_FP16;
    ثابت صحيح nloe = انگ_elems % VLEN_FP16;

    HVX_ويڪٽر جوڙ_sq_v = Q6_V_vsplat_R(0x00000000);
    HVX_ويڪٽر جوڙ_x_v  = Q6_V_vsplat_R(0x00000000);
    HVX_ويڪٽر epsilon_v = hvx_vec_splat_f32(epsilon);

    #هدايت unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ويڪٽر_Pair p = hvx_vec_f16_منزل_f32(v_src[i]);
        HVX_ويڪٽر p0 = Q6_V_lo_W(p);
        HVX_ويڪٽر p1 = Q6_V_hi_W(p);
        جوڙ_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(جوڙ_sq_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        جوڙ_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(جوڙ_sq_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
        جوڙ_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(جوڙ_x_v,  Q6_Vqf32_vadd_VsfVsf(p0, Q6_V_vzero()));
        جوڙ_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(جوڙ_x_v,  Q6_Vqf32_vadd_VsfVsf(p1, Q6_V_vzero()));
    }

    إذا (nloe > 0) {
        HVX_ويڪٽر_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_ويڪٽر v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ويڪٽر_Pair p = hvx_vec_f16_منزل_f32(v1);
        HVX_ويڪٽر p0 = Q6_V_lo_W(p);
        HVX_ويڪٽر p1 = Q6_V_hi_W(p);
        جوڙ_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(جوڙ_sq_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        جوڙ_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(جوڙ_sq_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
        جوڙ_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(جوڙ_x_v,  Q6_Vqf32_vadd_VsfVsf(p0, Q6_V_vzero()));
        جوڙ_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(جوڙ_x_v,  Q6_Vqf32_vadd_VsfVsf(p1, Q6_V_vzero()));
    }

    جوڙ_sq_v = hvx_vec_reduce_جوڙ_f32(Q6_Vsf_equals_Vqf32(جوڙ_sq_v));
    جوڙ_x_v  = hvx_vec_reduce_جوڙ_f32(Q6_Vsf_equals_Vqf32(جوڙ_x_v));

    HVX_ويڪٽر t_v            = hvx_vec_splat_f32((عائم) انگ_elems);
    HVX_ويڪٽر denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_ويڪٽر سراسري_sq_v      = Q6_Vqf32_vmpy_VsfVsf(جوڙ_sq_v, denom_v);
    HVX_ويڪٽر سراسري_x_v       = Q6_Vqf32_vmpy_VsfVsf(جوڙ_x_v,  denom_v);
    HVX_ويڪٽر سراسري_x_sq_v    = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(سراسري_x_v), Q6_Vsf_equals_Vqf32(سراسري_x_v));
    HVX_ويڪٽر var_v          = Q6_Vqf32_vsub_Vqf32Vqf32(سراسري_sq_v, سراسري_x_sq_v);
    HVX_ويڪٽر var_epsilon_v  = Q6_Vqf32_vadd_Vqf32Vsf(var_v, epsilon_v);

    HVX_ويڪٽر scale_v  = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(var_epsilon_v));
    HVX_ويڪٽر سراسري_x_b = hvx_vec_repl_f32(Q6_Vsf_equals_Vqf32(سراسري_x_v));

    #هدايت unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ويڪٽر_Pair p = hvx_vec_f16_منزل_f32(v_src[i]);
        HVX_ويڪٽر d0 = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(p), سراسري_x_b);
        HVX_ويڪٽر d1 = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(p), سراسري_x_b);
        HVX_ويڪٽر r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d0), scale_v));
        HVX_ويڪٽر r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d1), scale_v));
        v_dst[i] = hvx_vec_f32_منزل_f16(r0, r1);
    }

    إذا (nloe > 0) {
        HVX_ويڪٽر_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_ويڪٽر v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ويڪٽر_Pair p = hvx_vec_f16_منزل_f32(v1);
        HVX_ويڪٽر d0 = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(p), سراسري_x_b);
        HVX_ويڪٽر d1 = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(p), سراسري_x_b);
        HVX_ويڪٽر r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d0), scale_v));
        HVX_ويڪٽر r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d1), scale_v));
        HVX_ويڪٽر نتيجو = hvx_vec_f32_منزل_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, نتيجو);
    }
}

ساكن مضمن فراغ hvx_fast_l2_norm_f16(ثابت uint8_t * محدود src,
                                        uint8_t * محدود dst,
                                        ثابت صحيح انگ_elems,
                                        عائم     epsilon) {

    ثابت HVX_ويڪٽر * محدود v_src = (HVX_ويڪٽر *) src;
    HVX_ويڪٽر * محدود v_dst       = (HVX_ويڪٽر *) dst;

    ثابت صحيح nvec = انگ_elems / VLEN_FP16;
    ثابت صحيح nloe = انگ_elems % VLEN_FP16;

    HVX_ويڪٽر جوڙ_v = hvx_vec_splat_f32(0.0f);

    #هدايت unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ويڪٽر_Pair p = hvx_vec_f16_منزل_f32(v_src[i]);
        HVX_ويڪٽر p0 = Q6_V_lo_W(p);
        HVX_ويڪٽر p1 = Q6_V_hi_W(p);
        جوڙ_v = Q6_Vqf32_vadd_Vqf32Vqf32(جوڙ_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        جوڙ_v = Q6_Vqf32_vadd_Vqf32Vqf32(جوڙ_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    إذا (nloe > 0) {
        HVX_ويڪٽر_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_ويڪٽر v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ويڪٽر_Pair p = hvx_vec_f16_منزل_f32(v1);
        HVX_ويڪٽر p0 = Q6_V_lo_W(p);
        HVX_ويڪٽر p1 = Q6_V_hi_W(p);
        جوڙ_v = Q6_Vqf32_vadd_Vqf32Vqf32(جوڙ_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        جوڙ_v = Q6_Vqf32_vadd_Vqf32Vqf32(جوڙ_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    HVX_ويڪٽر جوڙ_sf    = hvx_vec_reduce_جوڙ_f32(Q6_Vsf_equals_Vqf32(جوڙ_v));
    HVX_ويڪٽر rsqrt_v   = hvx_vec_rsqrt_f32(جوڙ_sf);
    HVX_ويڪٽر sqrt_v    = hvx_vec_inverse_f32(rsqrt_v);
    HVX_ويڪٽر epsilon_v = hvx_vec_splat_f32(epsilon);
    HVX_ويڪٽر denom_v   = Q6_Vsf_vmax_VsfVsf(sqrt_v, epsilon_v);
    HVX_ويڪٽر scale_v   = hvx_vec_inverse_f32(denom_v);

    #هدايت unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ويڪٽر_Pair p = hvx_vec_f16_منزل_f32(v_src[i]);
        HVX_ويڪٽر r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_ويڪٽر r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        v_dst[i] = hvx_vec_f32_منزل_f16(r0, r1);
    }

    إذا (nloe > 0) {
        HVX_ويڪٽر_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_ويڪٽر v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ويڪٽر_Pair p = hvx_vec_f16_منزل_f32(v1);
        HVX_ويڪٽر r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_ويڪٽر r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        HVX_ويڪٽر نتيجو = hvx_vec_f32_منزل_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, نتيجو);
    }
}

#جيڪڏهن_پورو // HVX_NORM_H
