#ifndef HVX_REDUCE_H
#define HVX_REDUCE_H

#अन्तर्भाव <math.h>
#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <assert.h>

#अन्तर्भाव "hex-utils.h"
#अन्तर्भाव "hvx-base.h"
#अन्तर्भाव "hvx-types.h"

स्थिर अन्तःस्थ HVX_Vector hvx_vec_reduce_sum_n_i32(HVX_Vector in, unsigned पूर्णाङ्क n) {
    unsigned पूर्णाङ्क total = n * 4;  // total vec nbytes
    unsigned पूर्णाङ्क width = 4;      // int32

    HVX_Vector sum = in, sum_t;
    याव (width < total) {
        sum_t = Q6_V_vror_VR(sum, width);     // rotate right
        sum   = Q6_Vw_vadd_VwVw(sum_t, sum);  // elementwise sum
        width = width << 1;
    }
    पटिनिवत्त sum;
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_reduce_sum_i32(HVX_Vector in) {
    पटिनिवत्त hvx_vec_reduce_sum_n_i32(in, 32);
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_reduce_sum_n_qf32(HVX_Vector in, unsigned पूर्णाङ्क n) {
    unsigned पूर्णाङ्क total = n * 4;  // total vec nbytes
    unsigned पूर्णाङ्क width = 4;      // fp32 nbytes

    HVX_Vector sum = in, sum_t;
    याव (width < total) {
        sum_t = Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum), width);  // rotate right
        sum   = Q6_Vqf32_vadd_Vqf32Vsf(sum, sum_t);             // elementwise sum
        width = width << 1;
    }
    पटिनिवत्त sum;
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_reduce_sum_qf32(HVX_Vector in) {
    पटिनिवत्त hvx_vec_reduce_sum_n_qf32(in, 32);
}

#सचे __HVX_ARCH__ > 75

स्थिर अन्तःस्थ HVX_Vector hvx_vec_reduce_sum_f32x4(HVX_Vector_x4 in) {
    HVX_VectorPair sum_p01 = Q6_W_vshuff_VVR(in.v[1], in.v[0], 4);
    HVX_VectorPair sum_p23 = Q6_W_vshuff_VVR(in.v[3], in.v[2], 4);
    HVX_Vector  sum_sf01  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sum_p01), Q6_V_hi_W(sum_p01));
    HVX_Vector  sum_sf23  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sum_p23), Q6_V_hi_W(sum_p23));

    HVX_VectorPair sum_p0123 = Q6_W_vshuff_VVR(sum_sf23, sum_sf01, 8);
    HVX_Vector  sum_sf       = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sum_p0123), Q6_V_hi_W(sum_p0123));

    sum_sf = Q6_Vsf_vadd_VsfVsf(sum_sf, Q6_V_vror_VR(sum_sf, VLEN / 2));
    sum_sf = Q6_Vsf_vadd_VsfVsf(sum_sf, Q6_V_vror_VR(sum_sf, VLEN / 4));
    sum_sf = Q6_Vsf_vadd_VsfVsf(sum_sf, Q6_V_vror_VR(sum_sf, VLEN / 8));
    पटिनिवत्त sum_sf;
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_reduce_sum_f32x2(HVX_Vector in0, HVX_Vector in1) {
    HVX_VectorPair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_Vector  sum_sf  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    sum_sf = Q6_Vsf_vadd_VsfVsf(sum_sf, Q6_V_vror_VR(sum_sf, VLEN / 2));
    sum_sf = Q6_Vsf_vadd_VsfVsf(sum_sf, Q6_V_vror_VR(sum_sf, VLEN / 4));
    sum_sf = Q6_Vsf_vadd_VsfVsf(sum_sf, Q6_V_vror_VR(sum_sf, VLEN / 8));
    sum_sf = Q6_Vsf_vadd_VsfVsf(sum_sf, Q6_V_vror_VR(sum_sf, VLEN / 16));
    पटिनिवत्त sum_sf;
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_reduce_sum_n_f32(HVX_Vector in, unsigned पूर्णाङ्क n) {
    unsigned पूर्णाङ्क total = n * 4;  // total vec nbytes
    unsigned पूर्णाङ्क width = 4;      // fp32 nbytes

    HVX_Vector sum = in, sum_t;
    याव (width < total) {
        sum_t = Q6_V_vror_VR(sum, width);       // rotate right
        sum   = Q6_Vsf_vadd_VsfVsf(sum, sum_t); // elementwise sum
        width = width << 1;
    }
    पटिनिवत्त sum;
}

#अञ्ञथा

स्थिर अन्तःस्थ HVX_Vector hvx_vec_reduce_sum_f32x4(HVX_Vector_x4 in) {
    HVX_VectorPair sum_p01  = Q6_W_vshuff_VVR(in.v[1], in.v[0], 4);
    HVX_VectorPair sum_p23  = Q6_W_vshuff_VVR(in.v[3], in.v[2], 4);
    HVX_Vector     sum_qf01 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(sum_p01), Q6_V_hi_W(sum_p01));
    HVX_Vector     sum_qf23 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(sum_p23), Q6_V_hi_W(sum_p23));

    HVX_VectorPair sum_p0123 = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(sum_qf23), Q6_Vsf_equals_Vqf32(sum_qf01), 8);
    HVX_Vector     sum_qf    = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(sum_p0123), Q6_V_hi_W(sum_p0123));

    sum_qf = Q6_Vqf32_vadd_Vqf32Vsf(sum_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum_qf), VLEN / 2));
    sum_qf = Q6_Vqf32_vadd_Vqf32Vsf(sum_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum_qf), VLEN / 4));
    sum_qf = Q6_Vqf32_vadd_Vqf32Vsf(sum_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum_qf), VLEN / 8));
    पटिनिवत्त Q6_Vsf_equals_Vqf32(sum_qf);
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_reduce_sum_f32x2(HVX_Vector in0, HVX_Vector in1) {
    HVX_VectorPair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_Vector  sum_qf  = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    sum_qf = Q6_Vqf32_vadd_Vqf32Vsf(sum_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum_qf), VLEN / 2));
    sum_qf = Q6_Vqf32_vadd_Vqf32Vsf(sum_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum_qf), VLEN / 4));
    sum_qf = Q6_Vqf32_vadd_Vqf32Vsf(sum_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum_qf), VLEN / 8));
    sum_qf = Q6_Vqf32_vadd_Vqf32Vsf(sum_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum_qf), VLEN / 16));
    पटिनिवत्त Q6_Vsf_equals_Vqf32(sum_qf);
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_reduce_sum_n_f32(HVX_Vector in, unsigned पूर्णाङ्क n) {
    unsigned पूर्णाङ्क total = n * 4;  // total vec nbytes
    unsigned पूर्णाङ्क width = 4;      // fp32 nbytes

    HVX_Vector sum = in, sum_t;
    याव (width < total) {
        sum_t = Q6_V_vror_VR(sum, width);                               // rotate right
        sum   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(sum, sum_t));  // elementwise sum
        width = width << 1;
    }
    पटिनिवत्त sum;
}

#endif

स्थिर अन्तःस्थ HVX_Vector hvx_vec_reduce_sum_f32(HVX_Vector in) {
    पटिनिवत्त hvx_vec_reduce_sum_n_f32(in, 32);
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_reduce_max_f16(HVX_Vector in) {
    unsigned total = 128;  // total vec nbytes
    unsigned width = 2;    // fp16 nbytes

    HVX_Vector _max = in, _max_t;
    याव (width < total) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    पटिनिवत्त _max;
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_reduce_max2_f16(HVX_Vector in, HVX_Vector _max) {
    unsigned total = 128;  // total vec nbytes
    unsigned width = 2;    // fp32 nbytes

    HVX_Vector _max_t;

    _max = Q6_Vhf_vmax_VhfVhf(in, _max);
    याव (width < total) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    पटिनिवत्त _max;
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_reduce_max_f32(HVX_Vector in) {
    unsigned total = 128;  // total vec nbytes
    unsigned width = 4;    // fp32 nbytes

    HVX_Vector _max = in, _max_t;
    याव (width < total) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    पटिनिवत्त _max;
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_reduce_max2_f32(HVX_Vector in, HVX_Vector _max) {
    unsigned total = 128;  // total vec nbytes
    unsigned width = 4;    // fp32 nbytes

    HVX_Vector _max_t;

    _max = Q6_Vsf_vmax_VsfVsf(in, _max);
    याव (width < total) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    पटिनिवत्त _max;
}

#define hvx_reduce_loop_body(src_type, init_vec, pad_vec, vec_op, reduce_op, scalar_reduce) \
    करोहि {                                                                                    \
        src_type * restrict vsrc = (src_type *) src;                                        \
        HVX_Vector acc = init_vec;                                                          \
                                                                                            \
        ध्रुव uint32_t elem_size = परिमाण(भिन्न);                                           \
        ध्रुव uint32_t epv  = 128 / elem_size;                                              \
        ध्रुव uint32_t nvec = अङ्कः_elems / epv;                                              \
        ध्रुव uint32_t nloe = अङ्कः_elems % epv;                                              \
                                                                                            \
        uint32_t i = 0;                                                                     \
        _Pragma("unroll(4)")                                                                \
        अत्थाय (; i < nvec; i++) {                                                             \
            acc = vec_op(acc, vsrc[i]);                                                     \
        }                                                                                   \
        सचे (nloe) {                                                                         \
            ध्रुव भिन्न * srcf = (ध्रुव भिन्न *) src + i * epv;                             \
            HVX_Vector in = *(HVX_UVector *) srcf;                                          \
            HVX_Vector temp = Q6_V_valign_VVR(in, pad_vec, nloe * elem_size);               \
            acc = vec_op(acc, temp);                                                        \
        }                                                                                   \
        HVX_Vector v = reduce_op(acc);                                                      \
        पटिनिवत्त scalar_reduce(v);                                                            \
    } याव(0)

#define HVX_REDUCE_MAX_OP(acc, val) Q6_Vsf_vmax_VsfVsf(acc, val)
#define HVX_REDUCE_SUM_OP(acc, val) Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(acc), val)
#define HVX_SUM_SQ_OP(acc, val) Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(val, val))
#define HVX_REDUCE_MAX_SCALAR(v) hvx_vec_get_f32(v)
#define HVX_REDUCE_SUM_SCALAR(v) hvx_vec_get_f32(Q6_Vsf_equals_Vqf32(v))

// Max variants

स्थिर अन्तःस्थ भिन्न hvx_reduce_max_f32_a(ध्रुव uint8_t * restrict src, ध्रुव पूर्णाङ्क अङ्कः_elems) {
    HVX_Vector init_vec = hvx_vec_splat_f32(((ध्रुव भिन्न *) src)[0]);
    प्रतिज्ञा((unsigned long) src % 128 == 0);
    hvx_reduce_loop_body(HVX_Vector, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

स्थिर अन्तःस्थ भिन्न hvx_reduce_max_f32_u(ध्रुव uint8_t * restrict src, ध्रुव पूर्णाङ्क अङ्कः_elems) {
    HVX_Vector init_vec = hvx_vec_splat_f32(((ध्रुव भिन्न *) src)[0]);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

स्थिर अन्तःस्थ भिन्न hvx_reduce_max_f32(ध्रुव uint8_t * restrict src, ध्रुव पूर्णाङ्क अङ्कः_elems) {
    सचे (hex_is_aligned((शून्य *) src, 128)) {
        पटिनिवत्त hvx_reduce_max_f32_a(src, अङ्कः_elems);
    } अञ्ञथा {
        पटिनिवत्त hvx_reduce_max_f32_u(src, अङ्कः_elems);
    }
}

// Sum variants

स्थिर अन्तःस्थ भिन्न hvx_reduce_sum_f32_a(ध्रुव uint8_t * restrict src, ध्रुव पूर्णाङ्क अङ्कः_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(0);
    प्रतिज्ञा((unsigned long) src % 128 == 0);
    hvx_reduce_loop_body(HVX_Vector, init_vec, init_vec, HVX_REDUCE_SUM_OP, hvx_vec_reduce_sum_qf32, HVX_REDUCE_SUM_SCALAR);
}

स्थिर अन्तःस्थ भिन्न hvx_reduce_sum_f32_u(ध्रुव uint8_t * restrict src, ध्रुव पूर्णाङ्क अङ्कः_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_SUM_OP, hvx_vec_reduce_sum_qf32, HVX_REDUCE_SUM_SCALAR);
}

स्थिर अन्तःस्थ भिन्न hvx_reduce_sum_f32(ध्रुव uint8_t * restrict src, ध्रुव पूर्णाङ्क अङ्कः_elems) {
    सचे (hex_is_aligned((शून्य *) src, 128)) {
        पटिनिवत्त hvx_reduce_sum_f32_a(src, अङ्कः_elems);
    } अञ्ञथा {
        पटिनिवत्त hvx_reduce_sum_f32_u(src, अङ्कः_elems);
    }
}

// Sum of squares variants

स्थिर अन्तःस्थ भिन्न hvx_sum_of_squares_f32_a(ध्रुव uint8_t * restrict src, ध्रुव पूर्णाङ्क अङ्कः_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(0);
    प्रतिज्ञा((uintptr_t) src % 128 == 0);
    hvx_reduce_loop_body(HVX_Vector, init_vec, init_vec, HVX_SUM_SQ_OP, hvx_vec_reduce_sum_qf32, HVX_REDUCE_SUM_SCALAR);
}

स्थिर अन्तःस्थ भिन्न hvx_sum_of_squares_f32_u(ध्रुव uint8_t * restrict src, ध्रुव पूर्णाङ्क अङ्कः_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_SUM_SQ_OP, hvx_vec_reduce_sum_qf32, HVX_REDUCE_SUM_SCALAR);
}

स्थिर अन्तःस्थ भिन्न hvx_sum_of_squares_f32(ध्रुव uint8_t * restrict src, ध्रुव पूर्णाङ्क अङ्कः_elems) {
    सचे (hex_is_aligned((शून्य *) src, 128)) {
        पटिनिवत्त hvx_sum_of_squares_f32_a(src, अङ्कः_elems);
    } अञ्ञथा {
        पटिनिवत्त hvx_sum_of_squares_f32_u(src, अङ्कः_elems);
    }
}

// Signed 32-bit Integer Max variants

स्थिर अन्तःस्थ HVX_Vector hvx_vec_reduce_max_n_i32(HVX_Vector in, unsigned पूर्णाङ्क n) {
    unsigned पूर्णाङ्क total = n * 4;  // total vec nbytes
    unsigned पूर्णाङ्क width = 4;      // int32 nbytes

    HVX_Vector max_val = in, max_t;
    याव (width < total) {
        max_t = Q6_V_vror_VR(max_val, width);          // rotate right
        max_val = Q6_Vw_vmax_VwVw(max_t, max_val);      // elementwise signed max
        width = width << 1;
    }
    पटिनिवत्त max_val;
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_reduce_max_i32(HVX_Vector in) {
    पटिनिवत्त hvx_vec_reduce_max_n_i32(in, 32);
}

स्थिर अन्तःस्थ int32_t hvx_reduce_max_i32_a(ध्रुव uint8_t * restrict src, ध्रुव पूर्णाङ्क अङ्कः_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(((ध्रुव int32_t *) src)[0]);
    HVX_Vector pad_vec  = Q6_V_vsplat_R(0x80000000);
    प्रतिज्ञा((uintptr_t) src % 128 == 0);
    hvx_reduce_loop_body(HVX_Vector, init_vec, pad_vec, Q6_Vw_vmax_VwVw, hvx_vec_reduce_max_i32, hvx_vec_get_i32);
}

स्थिर अन्तःस्थ int32_t hvx_reduce_max_i32_u(ध्रुव uint8_t * restrict src, ध्रुव पूर्णाङ्क अङ्कः_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(((ध्रुव int32_t *) src)[0]);
    HVX_Vector pad_vec  = Q6_V_vsplat_R(0x80000000);
    hvx_reduce_loop_body(HVX_UVector, init_vec, pad_vec, Q6_Vw_vmax_VwVw, hvx_vec_reduce_max_i32, hvx_vec_get_i32);
}

स्थिर अन्तःस्थ int32_t hvx_reduce_max_i32(ध्रुव uint8_t * restrict src, ध्रुव पूर्णाङ्क अङ्कः_elems) {
    सचे (hex_is_aligned((शून्य *) src, 128)) {
        पटिनिवत्त hvx_reduce_max_i32_a(src, अङ्कः_elems);
    } अञ्ञथा {
        पटिनिवत्त hvx_reduce_max_i32_u(src, अङ्कः_elems);
    }
}

स्थिर अन्तःस्थ शून्य hvx_argmax_f32(
    ध्रुव भिन्न * restrict src,
    uint32_t n,
    uint32_t offset,
    भिन्न * out_val,
    int32_t * out_idx
) {
    सचे (n == 0) {
        *out_val = -INFINITY;
        *out_idx = (int32_t) offset;
        पटिनिवत्त;
    }

    सचे (n < 32 || !hex_is_aligned((शून्य *) src, 128)) {
        भिन्न best_val = src[0];
        int32_t best_idx = (int32_t) offset;
        अत्थाय (uint32_t i = 1; i < n; i++) {
            सचे (src[i] > best_val) {
                best_val = src[i];
                best_idx = (int32_t) (offset + i);
            }
        }
        *out_val = best_val;
        *out_idx = best_idx;
        पटिनिवत्त;
    }

    स्थिर ध्रुव int32_t c_lane_idx[32] __attribute__((aligned(128))) = {
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    };

    ध्रुव HVX_Vector v_init_idx = *(ध्रुव HVX_Vector *) c_lane_idx;
    ध्रुव HVX_Vector v_step     = Q6_V_vsplat_R(32);
    HVX_Vector v_cur_idx        = Q6_Vw_vadd_VwVw(v_init_idx, Q6_V_vsplat_R((int32_t) offset));
    HVX_Vector v_max_val        = hvx_vec_splat_f32(-INFINITY);
    HVX_Vector v_max_idx        = v_cur_idx;

    ध्रुव HVX_Vector * vsrc = (ध्रुव HVX_Vector *) src;
    ध्रुव uint32_t nvec = n / 32;

    अत्थाय (uint32_t vi = 0; vi < nvec; vi++) {
        HVX_Vector v = vsrc[vi];
        HVX_VectorPred pred = Q6_Q_vcmp_gt_VsfVsf(v, v_max_val);
        v_max_val = Q6_V_vmux_QVV(pred, v, v_max_val);
        v_max_idx = Q6_V_vmux_QVV(pred, v_cur_idx, v_max_idx);
        v_cur_idx = Q6_Vw_vadd_VwVw(v_cur_idx, v_step);
    }

    HVX_VectorAlias u_val, u_idx;
    u_val.v = v_max_val;
    u_idx.v = v_max_idx;

    भिन्न best_val = u_val.fp32[0];
    int32_t best_idx = (int32_t) u_idx.w[0];
    अत्थाय (पूर्णाङ्क i = 1; i < 32; i++) {
        सचे (u_val.fp32[i] > best_val) {
            best_val = u_val.fp32[i];
            best_idx = (int32_t) u_idx.w[i];
        }
    }

    अत्थाय (uint32_t i = nvec * 32; i < n; i++) {
        सचे (src[i] > best_val) {
            best_val = src[i];
            best_idx = (int32_t) (offset + i);
        }
    }

    *out_val = best_val;
    *out_idx = best_idx;
}

#undef hvx_reduce_loop_body
#undef HVX_REDUCE_MAX_OP
#undef HVX_REDUCE_SUM_OP
#undef HVX_REDUCE_MAX_SCALAR
#undef HVX_REDUCE_SUM_SCALAR
#undef HVX_SUM_SQ_OP

#endif /* HVX_REDUCE_H */
