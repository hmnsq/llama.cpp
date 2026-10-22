#अगर_अघोषित HVX_NORM_H
#घोषणा HVX_NORM_H

#समावेश <stdint.h>
#समावेश "hvx-base.h"
#समावेश "hvx-reduce.h"
#समावेश "hvx-inverse.h"
#समावेश "hvx-sqrt.h"
#समावेश "hvx-repl.h"

अटल अंतरभूत व्योम hvx_fast_rms_norm_f32(स्थिर uint8_t * restrict src,
                                         uint8_t * restrict dst,
                                         स्थिर पूर्णांक अंक_elems,
                                         भग्न     epsilon) {

    स्थिर HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    स्थिर पूर्णांक nvec = अंक_elems / VLEN_FP32;    // number of full vectors
    स्थिर पूर्णांक nloe = अंक_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares for full vectors
    HVX_Vector योग_v = Q6_V_vsplat_R(0x00000000);
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);

    #आशय unroll(4)
    क्रम (पूर्णांक i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        योग_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_v, v2);
    }

    // Handle tail elements using vectorized ops with masking
    अगर (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        योग_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_v, v2);
    }

    // Reduce HVX sum
    योग_v = hvx_vec_reduce_योग_f32(Q6_Vsf_equals_Vqf32(योग_v));

    HVX_Vector t_v            = hvx_vec_splat_f32((भग्न) अंक_elems);
    HVX_Vector denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_Vector mean_v         = Q6_Vqf32_vmpy_VsfVsf(योग_v, denom_v);
    HVX_Vector mean_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(mean_v, epsilon_v);

    // Scale full vectors
    HVX_Vector scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(mean_epsilon_v));

    #आशय unroll(4)
    क्रम (पूर्णांक i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        v_dst[i] = Q6_Vsf_equals_Vqf32(v2);
    }

    // Handle tail elements using vectorized ops with masking
    अगर (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_Vector परिणाम = Q6_Vsf_equals_Vqf32(v2);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, परिणाम);
    }
}

अटल अंतरभूत व्योम hvx_fast_rms_norm_mul_f32(स्थिर uint8_t * restrict src,
                                             स्थिर uint8_t * restrict weight,
                                             uint8_t * restrict dst,
                                             स्थिर पूर्णांक अंक_elems,
                                             भग्न     epsilon) {
    स्थिर HVX_Vector * restrict v_src    = (स्थिर HVX_Vector *) src;
    स्थिर HVX_Vector * restrict v_weight = (स्थिर HVX_Vector *) weight;
    HVX_Vector * restrict v_dst          = (HVX_Vector *) dst;

    स्थिर पूर्णांक nvec = अंक_elems / VLEN_FP32;    // number of full vectors
    स्थिर पूर्णांक nloe = अंक_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares for full vectors
    HVX_Vector योग_v = Q6_V_vsplat_R(0x00000000);
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);

    #आशय unroll(4)
    क्रम (पूर्णांक i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        योग_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_v, v2);
    }

    // Handle tail elements using vectorized ops with masking
    अगर (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        योग_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_v, v2);
    }

    // Reduce HVX sum
    योग_v = hvx_vec_reduce_योग_f32(Q6_Vsf_equals_Vqf32(योग_v));

    HVX_Vector t_v            = hvx_vec_splat_f32((भग्न) अंक_elems);
    HVX_Vector denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_Vector mean_v         = Q6_Vqf32_vmpy_VsfVsf(योग_v, denom_v);
    HVX_Vector mean_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(mean_v, epsilon_v);

    // Scale and multiply
    HVX_Vector scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(mean_epsilon_v));

    #आशय unroll(4)
    क्रम (पूर्णांक i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_Vector v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_Vector परिणाम = Q6_Vqf32_vmpy_VsfVsf(v3, v_weight[i]);
        v_dst[i] = Q6_Vsf_equals_Vqf32(परिणाम);
    }

    // Handle tail elements using vectorized ops with masking
    अगर (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_Vector v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_Vector परिणाम = Q6_Vqf32_vmpy_VsfVsf(v3, v_weight[nvec]);
        HVX_Vector res_v = Q6_Vsf_equals_Vqf32(परिणाम);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, res_v);
    }
}

अटल अंतरभूत व्योम hvx_fast_norm_f32(स्थिर uint8_t * restrict src,
                                     uint8_t * restrict dst,
                                     स्थिर पूर्णांक अंक_elems,
                                     भग्न     epsilon) {

    स्थिर HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    स्थिर पूर्णांक nvec = अंक_elems / VLEN_FP32;    // number of full vectors
    स्थिर पूर्णांक nloe = अंक_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares and sum of values for full vectors
    HVX_Vector योग_sq_v = Q6_V_vsplat_R(0x00000000);
    HVX_Vector योग_x_v  = Q6_V_vsplat_R(0x00000000);
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);

    #आशय unroll(4)
    क्रम (पूर्णांक i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        योग_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_sq_v, v2);
        योग_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(योग_x_v,  Q6_Vqf32_vadd_VsfVsf(v1, Q6_V_vzero()));
    }

    // Handle tail elements using vectorized ops with masking
    अगर (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        योग_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_sq_v, v2);
        योग_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(योग_x_v,  Q6_Vqf32_vadd_VsfVsf(v1, Q6_V_vzero()));
    }

    // Reduce HVX sums
    योग_sq_v = hvx_vec_reduce_योग_f32(Q6_Vsf_equals_Vqf32(योग_sq_v));
    योग_x_v  = hvx_vec_reduce_योग_f32(Q6_Vsf_equals_Vqf32(योग_x_v));

    HVX_Vector t_v            = hvx_vec_splat_f32((भग्न) अंक_elems);
    HVX_Vector denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_Vector mean_sq_v      = Q6_Vqf32_vmpy_VsfVsf(योग_sq_v, denom_v);
    HVX_Vector mean_x_v       = Q6_Vqf32_vmpy_VsfVsf(योग_x_v,  denom_v);
    HVX_Vector mean_x_sq_v    = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(mean_x_v), Q6_Vsf_equals_Vqf32(mean_x_v));
    HVX_Vector var_v          = Q6_Vqf32_vsub_Vqf32Vqf32(mean_sq_v, mean_x_sq_v);
    HVX_Vector var_epsilon_v  = Q6_Vqf32_vadd_Vqf32Vsf(var_v, epsilon_v);

    // scale = rsqrt(variance + epsilon),  mean_x broadcast for subtraction
    HVX_Vector scale_v  = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(var_epsilon_v));
    HVX_Vector mean_x_b = hvx_vec_repl_f32(Q6_Vsf_equals_Vqf32(mean_x_v));

    #आशय unroll(4)
    क्रम (पूर्णांक i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vsub_VsfVsf(v1, mean_x_b);
        HVX_Vector v3 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v2), scale_v);
        v_dst[i] = Q6_Vsf_equals_Vqf32(v3);
    }

    // Handle tail elements using vectorized ops with masking
    अगर (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vsub_VsfVsf(v1, mean_x_b);
        HVX_Vector v3 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v2), scale_v);
        HVX_Vector परिणाम = Q6_Vsf_equals_Vqf32(v3);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, परिणाम);
    }
}

अटल अंतरभूत व्योम hvx_fast_l2_norm_f32(स्थिर uint8_t * restrict src,
                                        uint8_t * restrict dst,
                                        स्थिर पूर्णांक अंक_elems,
                                        भग्न     epsilon) {

    स्थिर HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    HVX_Vector योग_v = hvx_vec_splat_f32(0.0f);

    स्थिर पूर्णांक nvec = अंक_elems / VLEN_FP32;
    स्थिर पूर्णांक nloe = अंक_elems % VLEN_FP32;

    #आशय unroll(4)
    क्रम (पूर्णांक i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector sq = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        योग_v         = Q6_Vqf32_vadd_Vqf32Vqf32(योग_v, sq);
    }

    // Include tail elements in the sum-of-squares using a predicate mask
    अगर (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector sq = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        योग_v         = Q6_Vqf32_vadd_Vqf32Vqf32(योग_v, sq);
    }

    // Compute scale = 1/fmax(sqrt(sum), epsilon) entirely in HVX registers.
    // hvx_vec_rsqrt_f32 + hvx_vec_inverse_f32 avoids scalar extraction.
    HVX_Vector योग_sf    = hvx_vec_reduce_योग_f32(Q6_Vsf_equals_Vqf32(योग_v));
    HVX_Vector rsqrt_v   = hvx_vec_rsqrt_f32(योग_sf);              // 1/sqrt(sum)
    HVX_Vector sqrt_v    = hvx_vec_inverse_f32(rsqrt_v);            // sqrt(sum)
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);
    HVX_Vector denom_v   = Q6_Vsf_vmax_VsfVsf(sqrt_v, epsilon_v);  // fmax(sqrt(sum), epsilon)
    HVX_Vector scale_v   = hvx_vec_inverse_f32(denom_v);            // 1/fmax(sqrt(sum), epsilon)

    #आशय unroll(4)
    क्रम (पूर्णांक i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        v_dst[i]      = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, scale_v));
    }

    अगर (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector परिणाम = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, scale_v));
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, परिणाम);
    }
}

// F16 norm kernels: reduce and scale in f32 (via promote/narrow), matching the
// precision-preserving pattern used by the flash-attn f16 kernels.

अटल अंतरभूत व्योम hvx_fast_rms_norm_f16(स्थिर uint8_t * restrict src,
                                         uint8_t * restrict dst,
                                         स्थिर पूर्णांक अंक_elems,
                                         भग्न     epsilon) {

    स्थिर HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    स्थिर पूर्णांक nvec = अंक_elems / VLEN_FP16;    // number of full f16 vectors
    स्थिर पूर्णांक nloe = अंक_elems % VLEN_FP16;    // leftover elements

    HVX_Vector योग_v = Q6_V_vsplat_R(0x00000000);
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);

    #आशय unroll(4)
    क्रम (पूर्णांक i = 0; i < nvec; i++) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_src[i]);
        HVX_Vector p0 = Q6_V_lo_W(p);
        HVX_Vector p1 = Q6_V_hi_W(p);
        योग_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        योग_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    अगर (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_VectorPair p = hvx_vec_f16_to_f32(v1);
        HVX_Vector p0 = Q6_V_lo_W(p);
        HVX_Vector p1 = Q6_V_hi_W(p);
        योग_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        योग_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    योग_v = hvx_vec_reduce_योग_f32(Q6_Vsf_equals_Vqf32(योग_v));

    HVX_Vector t_v            = hvx_vec_splat_f32((भग्न) अंक_elems);
    HVX_Vector denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_Vector mean_v         = Q6_Vqf32_vmpy_VsfVsf(योग_v, denom_v);
    HVX_Vector mean_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(mean_v, epsilon_v);

    HVX_Vector scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(mean_epsilon_v));

    #आशय unroll(4)
    क्रम (पूर्णांक i = 0; i < nvec; i++) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_src[i]);
        HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        v_dst[i] = hvx_vec_f32_to_f16(r0, r1);
    }

    अगर (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_VectorPair p = hvx_vec_f16_to_f32(v1);
        HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        HVX_Vector परिणाम = hvx_vec_f32_to_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, परिणाम);
    }
}

अटल अंतरभूत व्योम hvx_fast_norm_f16(स्थिर uint8_t * restrict src,
                                     uint8_t * restrict dst,
                                     स्थिर पूर्णांक अंक_elems,
                                     भग्न     epsilon) {

    स्थिर HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    स्थिर पूर्णांक nvec = अंक_elems / VLEN_FP16;
    स्थिर पूर्णांक nloe = अंक_elems % VLEN_FP16;

    HVX_Vector योग_sq_v = Q6_V_vsplat_R(0x00000000);
    HVX_Vector योग_x_v  = Q6_V_vsplat_R(0x00000000);
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);

    #आशय unroll(4)
    क्रम (पूर्णांक i = 0; i < nvec; i++) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_src[i]);
        HVX_Vector p0 = Q6_V_lo_W(p);
        HVX_Vector p1 = Q6_V_hi_W(p);
        योग_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_sq_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        योग_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_sq_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
        योग_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(योग_x_v,  Q6_Vqf32_vadd_VsfVsf(p0, Q6_V_vzero()));
        योग_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(योग_x_v,  Q6_Vqf32_vadd_VsfVsf(p1, Q6_V_vzero()));
    }

    अगर (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_VectorPair p = hvx_vec_f16_to_f32(v1);
        HVX_Vector p0 = Q6_V_lo_W(p);
        HVX_Vector p1 = Q6_V_hi_W(p);
        योग_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_sq_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        योग_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_sq_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
        योग_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(योग_x_v,  Q6_Vqf32_vadd_VsfVsf(p0, Q6_V_vzero()));
        योग_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(योग_x_v,  Q6_Vqf32_vadd_VsfVsf(p1, Q6_V_vzero()));
    }

    योग_sq_v = hvx_vec_reduce_योग_f32(Q6_Vsf_equals_Vqf32(योग_sq_v));
    योग_x_v  = hvx_vec_reduce_योग_f32(Q6_Vsf_equals_Vqf32(योग_x_v));

    HVX_Vector t_v            = hvx_vec_splat_f32((भग्न) अंक_elems);
    HVX_Vector denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_Vector mean_sq_v      = Q6_Vqf32_vmpy_VsfVsf(योग_sq_v, denom_v);
    HVX_Vector mean_x_v       = Q6_Vqf32_vmpy_VsfVsf(योग_x_v,  denom_v);
    HVX_Vector mean_x_sq_v    = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(mean_x_v), Q6_Vsf_equals_Vqf32(mean_x_v));
    HVX_Vector var_v          = Q6_Vqf32_vsub_Vqf32Vqf32(mean_sq_v, mean_x_sq_v);
    HVX_Vector var_epsilon_v  = Q6_Vqf32_vadd_Vqf32Vsf(var_v, epsilon_v);

    HVX_Vector scale_v  = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(var_epsilon_v));
    HVX_Vector mean_x_b = hvx_vec_repl_f32(Q6_Vsf_equals_Vqf32(mean_x_v));

    #आशय unroll(4)
    क्रम (पूर्णांक i = 0; i < nvec; i++) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_src[i]);
        HVX_Vector d0 = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(p), mean_x_b);
        HVX_Vector d1 = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(p), mean_x_b);
        HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d0), scale_v));
        HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d1), scale_v));
        v_dst[i] = hvx_vec_f32_to_f16(r0, r1);
    }

    अगर (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_VectorPair p = hvx_vec_f16_to_f32(v1);
        HVX_Vector d0 = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(p), mean_x_b);
        HVX_Vector d1 = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(p), mean_x_b);
        HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d0), scale_v));
        HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d1), scale_v));
        HVX_Vector परिणाम = hvx_vec_f32_to_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, परिणाम);
    }
}

अटल अंतरभूत व्योम hvx_fast_l2_norm_f16(स्थिर uint8_t * restrict src,
                                        uint8_t * restrict dst,
                                        स्थिर पूर्णांक अंक_elems,
                                        भग्न     epsilon) {

    स्थिर HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    स्थिर पूर्णांक nvec = अंक_elems / VLEN_FP16;
    स्थिर पूर्णांक nloe = अंक_elems % VLEN_FP16;

    HVX_Vector योग_v = hvx_vec_splat_f32(0.0f);

    #आशय unroll(4)
    क्रम (पूर्णांक i = 0; i < nvec; i++) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_src[i]);
        HVX_Vector p0 = Q6_V_lo_W(p);
        HVX_Vector p1 = Q6_V_hi_W(p);
        योग_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        योग_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    अगर (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_VectorPair p = hvx_vec_f16_to_f32(v1);
        HVX_Vector p0 = Q6_V_lo_W(p);
        HVX_Vector p1 = Q6_V_hi_W(p);
        योग_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        योग_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    HVX_Vector योग_sf    = hvx_vec_reduce_योग_f32(Q6_Vsf_equals_Vqf32(योग_v));
    HVX_Vector rsqrt_v   = hvx_vec_rsqrt_f32(योग_sf);
    HVX_Vector sqrt_v    = hvx_vec_inverse_f32(rsqrt_v);
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);
    HVX_Vector denom_v   = Q6_Vsf_vmax_VsfVsf(sqrt_v, epsilon_v);
    HVX_Vector scale_v   = hvx_vec_inverse_f32(denom_v);

    #आशय unroll(4)
    क्रम (पूर्णांक i = 0; i < nvec; i++) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_src[i]);
        HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        v_dst[i] = hvx_vec_f32_to_f16(r0, r1);
    }

    अगर (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_VectorPair p = hvx_vec_f16_to_f32(v1);
        HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        HVX_Vector परिणाम = hvx_vec_f32_to_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, परिणाम);
    }
}

#पूर्ण_अगर // HVX_NORM_H
