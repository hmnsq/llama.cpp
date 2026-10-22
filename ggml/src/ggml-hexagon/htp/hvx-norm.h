#ifndef HVX_NORM_H
#define HVX_NORM_H

#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব "hvx-base.h"
#অন্তর্ভাব "hvx-reduce.h"
#অন্তর্ভাব "hvx-inverse.h"
#অন্তর্ভাব "hvx-sqrt.h"
#অন্তর্ভাব "hvx-repl.h"

স্থির অন্তঃরেখ শূন্য hvx_fast_rms_norm_f32(ধ্রুবক uint8_t * restrict src,
                                         uint8_t * restrict dst,
                                         ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems,
                                         দশমিক     epsilon) {

    ধ্রুবক HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    ধ্রুবক পূর্ণসংখ্যা nvec = অঙ্ক_elems / VLEN_FP32;    // number of full vectors
    ধ্রুবক পূর্ণসংখ্যা nloe = অঙ্ক_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares for full vectors
    HVX_Vector যোগ_v = Q6_V_vsplat_R(0x00000000);
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);

    #pragma unroll(4)
    জন্য (পূর্ণসংখ্যা i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        যোগ_v = Q6_Vqf32_vadd_Vqf32Vqf32(যোগ_v, v2);
    }

    // Handle tail elements using vectorized ops with masking
    যদি (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        যোগ_v = Q6_Vqf32_vadd_Vqf32Vqf32(যোগ_v, v2);
    }

    // Reduce HVX sum
    যোগ_v = hvx_vec_reduce_যোগ_f32(Q6_Vsf_equals_Vqf32(যোগ_v));

    HVX_Vector t_v            = hvx_vec_splat_f32((দশমিক) অঙ্ক_elems);
    HVX_Vector denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_Vector mean_v         = Q6_Vqf32_vmpy_VsfVsf(যোগ_v, denom_v);
    HVX_Vector mean_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(mean_v, epsilon_v);

    // Scale full vectors
    HVX_Vector scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(mean_epsilon_v));

    #pragma unroll(4)
    জন্য (পূর্ণসংখ্যা i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        v_dst[i] = Q6_Vsf_equals_Vqf32(v2);
    }

    // Handle tail elements using vectorized ops with masking
    যদি (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_Vector ফলাফল = Q6_Vsf_equals_Vqf32(v2);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, ফলাফল);
    }
}

স্থির অন্তঃরেখ শূন্য hvx_fast_rms_norm_mul_f32(ধ্রুবক uint8_t * restrict src,
                                             ধ্রুবক uint8_t * restrict weight,
                                             uint8_t * restrict dst,
                                             ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems,
                                             দশমিক     epsilon) {
    ধ্রুবক HVX_Vector * restrict v_src    = (ধ্রুবক HVX_Vector *) src;
    ধ্রুবক HVX_Vector * restrict v_weight = (ধ্রুবক HVX_Vector *) weight;
    HVX_Vector * restrict v_dst          = (HVX_Vector *) dst;

    ধ্রুবক পূর্ণসংখ্যা nvec = অঙ্ক_elems / VLEN_FP32;    // number of full vectors
    ধ্রুবক পূর্ণসংখ্যা nloe = অঙ্ক_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares for full vectors
    HVX_Vector যোগ_v = Q6_V_vsplat_R(0x00000000);
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);

    #pragma unroll(4)
    জন্য (পূর্ণসংখ্যা i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        যোগ_v = Q6_Vqf32_vadd_Vqf32Vqf32(যোগ_v, v2);
    }

    // Handle tail elements using vectorized ops with masking
    যদি (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        যোগ_v = Q6_Vqf32_vadd_Vqf32Vqf32(যোগ_v, v2);
    }

    // Reduce HVX sum
    যোগ_v = hvx_vec_reduce_যোগ_f32(Q6_Vsf_equals_Vqf32(যোগ_v));

    HVX_Vector t_v            = hvx_vec_splat_f32((দশমিক) অঙ্ক_elems);
    HVX_Vector denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_Vector mean_v         = Q6_Vqf32_vmpy_VsfVsf(যোগ_v, denom_v);
    HVX_Vector mean_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(mean_v, epsilon_v);

    // Scale and multiply
    HVX_Vector scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(mean_epsilon_v));

    #pragma unroll(4)
    জন্য (পূর্ণসংখ্যা i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_Vector v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_Vector ফলাফল = Q6_Vqf32_vmpy_VsfVsf(v3, v_weight[i]);
        v_dst[i] = Q6_Vsf_equals_Vqf32(ফলাফল);
    }

    // Handle tail elements using vectorized ops with masking
    যদি (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_Vector v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_Vector ফলাফল = Q6_Vqf32_vmpy_VsfVsf(v3, v_weight[nvec]);
        HVX_Vector res_v = Q6_Vsf_equals_Vqf32(ফলাফল);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, res_v);
    }
}

স্থির অন্তঃরেখ শূন্য hvx_fast_norm_f32(ধ্রুবক uint8_t * restrict src,
                                     uint8_t * restrict dst,
                                     ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems,
                                     দশমিক     epsilon) {

    ধ্রুবক HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    ধ্রুবক পূর্ণসংখ্যা nvec = অঙ্ক_elems / VLEN_FP32;    // number of full vectors
    ধ্রুবক পূর্ণসংখ্যা nloe = অঙ্ক_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares and sum of values for full vectors
    HVX_Vector যোগ_sq_v = Q6_V_vsplat_R(0x00000000);
    HVX_Vector যোগ_x_v  = Q6_V_vsplat_R(0x00000000);
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);

    #pragma unroll(4)
    জন্য (পূর্ণসংখ্যা i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        যোগ_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(যোগ_sq_v, v2);
        যোগ_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(যোগ_x_v,  Q6_Vqf32_vadd_VsfVsf(v1, Q6_V_vzero()));
    }

    // Handle tail elements using vectorized ops with masking
    যদি (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        যোগ_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(যোগ_sq_v, v2);
        যোগ_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(যোগ_x_v,  Q6_Vqf32_vadd_VsfVsf(v1, Q6_V_vzero()));
    }

    // Reduce HVX sums
    যোগ_sq_v = hvx_vec_reduce_যোগ_f32(Q6_Vsf_equals_Vqf32(যোগ_sq_v));
    যোগ_x_v  = hvx_vec_reduce_যোগ_f32(Q6_Vsf_equals_Vqf32(যোগ_x_v));

    HVX_Vector t_v            = hvx_vec_splat_f32((দশমিক) অঙ্ক_elems);
    HVX_Vector denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_Vector mean_sq_v      = Q6_Vqf32_vmpy_VsfVsf(যোগ_sq_v, denom_v);
    HVX_Vector mean_x_v       = Q6_Vqf32_vmpy_VsfVsf(যোগ_x_v,  denom_v);
    HVX_Vector mean_x_sq_v    = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(mean_x_v), Q6_Vsf_equals_Vqf32(mean_x_v));
    HVX_Vector var_v          = Q6_Vqf32_vsub_Vqf32Vqf32(mean_sq_v, mean_x_sq_v);
    HVX_Vector var_epsilon_v  = Q6_Vqf32_vadd_Vqf32Vsf(var_v, epsilon_v);

    // scale = rsqrt(variance + epsilon),  mean_x broadcast for subtraction
    HVX_Vector scale_v  = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(var_epsilon_v));
    HVX_Vector mean_x_b = hvx_vec_repl_f32(Q6_Vsf_equals_Vqf32(mean_x_v));

    #pragma unroll(4)
    জন্য (পূর্ণসংখ্যা i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vsub_VsfVsf(v1, mean_x_b);
        HVX_Vector v3 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v2), scale_v);
        v_dst[i] = Q6_Vsf_equals_Vqf32(v3);
    }

    // Handle tail elements using vectorized ops with masking
    যদি (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vsub_VsfVsf(v1, mean_x_b);
        HVX_Vector v3 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v2), scale_v);
        HVX_Vector ফলাফল = Q6_Vsf_equals_Vqf32(v3);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, ফলাফল);
    }
}

স্থির অন্তঃরেখ শূন্য hvx_fast_l2_norm_f32(ধ্রুবক uint8_t * restrict src,
                                        uint8_t * restrict dst,
                                        ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems,
                                        দশমিক     epsilon) {

    ধ্রুবক HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    HVX_Vector যোগ_v = hvx_vec_splat_f32(0.0f);

    ধ্রুবক পূর্ণসংখ্যা nvec = অঙ্ক_elems / VLEN_FP32;
    ধ্রুবক পূর্ণসংখ্যা nloe = অঙ্ক_elems % VLEN_FP32;

    #pragma unroll(4)
    জন্য (পূর্ণসংখ্যা i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector sq = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        যোগ_v         = Q6_Vqf32_vadd_Vqf32Vqf32(যোগ_v, sq);
    }

    // Include tail elements in the sum-of-squares using a predicate mask
    যদি (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector sq = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        যোগ_v         = Q6_Vqf32_vadd_Vqf32Vqf32(যোগ_v, sq);
    }

    // Compute scale = 1/fmax(sqrt(sum), epsilon) entirely in HVX registers.
    // hvx_vec_rsqrt_f32 + hvx_vec_inverse_f32 avoids scalar extraction.
    HVX_Vector যোগ_sf    = hvx_vec_reduce_যোগ_f32(Q6_Vsf_equals_Vqf32(যোগ_v));
    HVX_Vector rsqrt_v   = hvx_vec_rsqrt_f32(যোগ_sf);              // 1/sqrt(sum)
    HVX_Vector sqrt_v    = hvx_vec_inverse_f32(rsqrt_v);            // sqrt(sum)
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);
    HVX_Vector denom_v   = Q6_Vsf_vmax_VsfVsf(sqrt_v, epsilon_v);  // fmax(sqrt(sum), epsilon)
    HVX_Vector scale_v   = hvx_vec_inverse_f32(denom_v);            // 1/fmax(sqrt(sum), epsilon)

    #pragma unroll(4)
    জন্য (পূর্ণসংখ্যা i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        v_dst[i]      = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, scale_v));
    }

    যদি (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector ফলাফল = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, scale_v));
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, ফলাফল);
    }
}

// F16 norm kernels: reduce and scale in f32 (via promote/narrow), matching the
// precision-preserving pattern used by the flash-attn f16 kernels.

স্থির অন্তঃরেখ শূন্য hvx_fast_rms_norm_f16(ধ্রুবক uint8_t * restrict src,
                                         uint8_t * restrict dst,
                                         ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems,
                                         দশমিক     epsilon) {

    ধ্রুবক HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    ধ্রুবক পূর্ণসংখ্যা nvec = অঙ্ক_elems / VLEN_FP16;    // number of full f16 vectors
    ধ্রুবক পূর্ণসংখ্যা nloe = অঙ্ক_elems % VLEN_FP16;    // leftover elements

    HVX_Vector যোগ_v = Q6_V_vsplat_R(0x00000000);
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);

    #pragma unroll(4)
    জন্য (পূর্ণসংখ্যা i = 0; i < nvec; i++) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_src[i]);
        HVX_Vector p0 = Q6_V_lo_W(p);
        HVX_Vector p1 = Q6_V_hi_W(p);
        যোগ_v = Q6_Vqf32_vadd_Vqf32Vqf32(যোগ_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        যোগ_v = Q6_Vqf32_vadd_Vqf32Vqf32(যোগ_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    যদি (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_VectorPair p = hvx_vec_f16_to_f32(v1);
        HVX_Vector p0 = Q6_V_lo_W(p);
        HVX_Vector p1 = Q6_V_hi_W(p);
        যোগ_v = Q6_Vqf32_vadd_Vqf32Vqf32(যোগ_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        যোগ_v = Q6_Vqf32_vadd_Vqf32Vqf32(যোগ_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    যোগ_v = hvx_vec_reduce_যোগ_f32(Q6_Vsf_equals_Vqf32(যোগ_v));

    HVX_Vector t_v            = hvx_vec_splat_f32((দশমিক) অঙ্ক_elems);
    HVX_Vector denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_Vector mean_v         = Q6_Vqf32_vmpy_VsfVsf(যোগ_v, denom_v);
    HVX_Vector mean_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(mean_v, epsilon_v);

    HVX_Vector scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(mean_epsilon_v));

    #pragma unroll(4)
    জন্য (পূর্ণসংখ্যা i = 0; i < nvec; i++) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_src[i]);
        HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        v_dst[i] = hvx_vec_f32_to_f16(r0, r1);
    }

    যদি (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_VectorPair p = hvx_vec_f16_to_f32(v1);
        HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        HVX_Vector ফলাফল = hvx_vec_f32_to_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, ফলাফল);
    }
}

স্থির অন্তঃরেখ শূন্য hvx_fast_norm_f16(ধ্রুবক uint8_t * restrict src,
                                     uint8_t * restrict dst,
                                     ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems,
                                     দশমিক     epsilon) {

    ধ্রুবক HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    ধ্রুবক পূর্ণসংখ্যা nvec = অঙ্ক_elems / VLEN_FP16;
    ধ্রুবক পূর্ণসংখ্যা nloe = অঙ্ক_elems % VLEN_FP16;

    HVX_Vector যোগ_sq_v = Q6_V_vsplat_R(0x00000000);
    HVX_Vector যোগ_x_v  = Q6_V_vsplat_R(0x00000000);
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);

    #pragma unroll(4)
    জন্য (পূর্ণসংখ্যা i = 0; i < nvec; i++) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_src[i]);
        HVX_Vector p0 = Q6_V_lo_W(p);
        HVX_Vector p1 = Q6_V_hi_W(p);
        যোগ_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(যোগ_sq_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        যোগ_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(যোগ_sq_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
        যোগ_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(যোগ_x_v,  Q6_Vqf32_vadd_VsfVsf(p0, Q6_V_vzero()));
        যোগ_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(যোগ_x_v,  Q6_Vqf32_vadd_VsfVsf(p1, Q6_V_vzero()));
    }

    যদি (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_VectorPair p = hvx_vec_f16_to_f32(v1);
        HVX_Vector p0 = Q6_V_lo_W(p);
        HVX_Vector p1 = Q6_V_hi_W(p);
        যোগ_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(যোগ_sq_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        যোগ_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(যোগ_sq_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
        যোগ_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(যোগ_x_v,  Q6_Vqf32_vadd_VsfVsf(p0, Q6_V_vzero()));
        যোগ_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(যোগ_x_v,  Q6_Vqf32_vadd_VsfVsf(p1, Q6_V_vzero()));
    }

    যোগ_sq_v = hvx_vec_reduce_যোগ_f32(Q6_Vsf_equals_Vqf32(যোগ_sq_v));
    যোগ_x_v  = hvx_vec_reduce_যোগ_f32(Q6_Vsf_equals_Vqf32(যোগ_x_v));

    HVX_Vector t_v            = hvx_vec_splat_f32((দশমিক) অঙ্ক_elems);
    HVX_Vector denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_Vector mean_sq_v      = Q6_Vqf32_vmpy_VsfVsf(যোগ_sq_v, denom_v);
    HVX_Vector mean_x_v       = Q6_Vqf32_vmpy_VsfVsf(যোগ_x_v,  denom_v);
    HVX_Vector mean_x_sq_v    = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(mean_x_v), Q6_Vsf_equals_Vqf32(mean_x_v));
    HVX_Vector var_v          = Q6_Vqf32_vsub_Vqf32Vqf32(mean_sq_v, mean_x_sq_v);
    HVX_Vector var_epsilon_v  = Q6_Vqf32_vadd_Vqf32Vsf(var_v, epsilon_v);

    HVX_Vector scale_v  = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(var_epsilon_v));
    HVX_Vector mean_x_b = hvx_vec_repl_f32(Q6_Vsf_equals_Vqf32(mean_x_v));

    #pragma unroll(4)
    জন্য (পূর্ণসংখ্যা i = 0; i < nvec; i++) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_src[i]);
        HVX_Vector d0 = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(p), mean_x_b);
        HVX_Vector d1 = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(p), mean_x_b);
        HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d0), scale_v));
        HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d1), scale_v));
        v_dst[i] = hvx_vec_f32_to_f16(r0, r1);
    }

    যদি (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_VectorPair p = hvx_vec_f16_to_f32(v1);
        HVX_Vector d0 = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(p), mean_x_b);
        HVX_Vector d1 = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(p), mean_x_b);
        HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d0), scale_v));
        HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d1), scale_v));
        HVX_Vector ফলাফল = hvx_vec_f32_to_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, ফলাফল);
    }
}

স্থির অন্তঃরেখ শূন্য hvx_fast_l2_norm_f16(ধ্রুবক uint8_t * restrict src,
                                        uint8_t * restrict dst,
                                        ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems,
                                        দশমিক     epsilon) {

    ধ্রুবক HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    ধ্রুবক পূর্ণসংখ্যা nvec = অঙ্ক_elems / VLEN_FP16;
    ধ্রুবক পূর্ণসংখ্যা nloe = অঙ্ক_elems % VLEN_FP16;

    HVX_Vector যোগ_v = hvx_vec_splat_f32(0.0f);

    #pragma unroll(4)
    জন্য (পূর্ণসংখ্যা i = 0; i < nvec; i++) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_src[i]);
        HVX_Vector p0 = Q6_V_lo_W(p);
        HVX_Vector p1 = Q6_V_hi_W(p);
        যোগ_v = Q6_Vqf32_vadd_Vqf32Vqf32(যোগ_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        যোগ_v = Q6_Vqf32_vadd_Vqf32Vqf32(যোগ_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    যদি (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_VectorPair p = hvx_vec_f16_to_f32(v1);
        HVX_Vector p0 = Q6_V_lo_W(p);
        HVX_Vector p1 = Q6_V_hi_W(p);
        যোগ_v = Q6_Vqf32_vadd_Vqf32Vqf32(যোগ_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        যোগ_v = Q6_Vqf32_vadd_Vqf32Vqf32(যোগ_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    HVX_Vector যোগ_sf    = hvx_vec_reduce_যোগ_f32(Q6_Vsf_equals_Vqf32(যোগ_v));
    HVX_Vector rsqrt_v   = hvx_vec_rsqrt_f32(যোগ_sf);
    HVX_Vector sqrt_v    = hvx_vec_inverse_f32(rsqrt_v);
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);
    HVX_Vector denom_v   = Q6_Vsf_vmax_VsfVsf(sqrt_v, epsilon_v);
    HVX_Vector scale_v   = hvx_vec_inverse_f32(denom_v);

    #pragma unroll(4)
    জন্য (পূর্ণসংখ্যা i = 0; i < nvec; i++) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_src[i]);
        HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        v_dst[i] = hvx_vec_f32_to_f16(r0, r1);
    }

    যদি (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_VectorPair p = hvx_vec_f16_to_f32(v1);
        HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        HVX_Vector ফলাফল = hvx_vec_f32_to_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, ফলাফল);
    }
}

#endif // HVX_NORM_H
