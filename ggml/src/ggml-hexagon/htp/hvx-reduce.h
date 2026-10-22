#ifndef HVX_REDUCE_H
#define HVX_REDUCE_H

#அந்தர்பாவ <math.h>
#அந்தர்பாவ <stdbool.h>
#அந்தர்பாவ <stdint.h>
#அந்தர்பாவ <assert.h>

#அந்தர்பாவ "hex-utils.h"
#அந்தர்பாவ "hvx-base.h"
#அந்தர்பாவ "hvx-types.h"

நிலையான உள்வரி HVX_Vector hvx_vec_reduce_sum_n_i32(HVX_Vector in, unsigned முழு n) {
    unsigned முழு total = n * 4;  // total vec nbytes
    unsigned முழு width = 4;      // int32

    HVX_Vector sum = in, sum_t;
    போது (width < total) {
        sum_t = Q6_V_vror_VR(sum, width);     // rotate right
        sum   = Q6_Vw_vadd_VwVw(sum_t, sum);  // elementwise sum
        width = width << 1;
    }
    திரும்பு sum;
}

நிலையான உள்வரி HVX_Vector hvx_vec_reduce_sum_i32(HVX_Vector in) {
    திரும்பு hvx_vec_reduce_sum_n_i32(in, 32);
}

நிலையான உள்வரி HVX_Vector hvx_vec_reduce_sum_n_qf32(HVX_Vector in, unsigned முழு n) {
    unsigned முழு total = n * 4;  // total vec nbytes
    unsigned முழு width = 4;      // fp32 nbytes

    HVX_Vector sum = in, sum_t;
    போது (width < total) {
        sum_t = Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum), width);  // rotate right
        sum   = Q6_Vqf32_vadd_Vqf32Vsf(sum, sum_t);             // elementwise sum
        width = width << 1;
    }
    திரும்பு sum;
}

நிலையான உள்வரி HVX_Vector hvx_vec_reduce_sum_qf32(HVX_Vector in) {
    திரும்பு hvx_vec_reduce_sum_n_qf32(in, 32);
}

#எனில் __HVX_ARCH__ > 75

நிலையான உள்வரி HVX_Vector hvx_vec_reduce_sum_f32x4(HVX_Vector_x4 in) {
    HVX_VectorPair sum_p01 = Q6_W_vshuff_VVR(in.v[1], in.v[0], 4);
    HVX_VectorPair sum_p23 = Q6_W_vshuff_VVR(in.v[3], in.v[2], 4);
    HVX_Vector  sum_sf01  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sum_p01), Q6_V_hi_W(sum_p01));
    HVX_Vector  sum_sf23  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sum_p23), Q6_V_hi_W(sum_p23));

    HVX_VectorPair sum_p0123 = Q6_W_vshuff_VVR(sum_sf23, sum_sf01, 8);
    HVX_Vector  sum_sf       = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sum_p0123), Q6_V_hi_W(sum_p0123));

    sum_sf = Q6_Vsf_vadd_VsfVsf(sum_sf, Q6_V_vror_VR(sum_sf, VLEN / 2));
    sum_sf = Q6_Vsf_vadd_VsfVsf(sum_sf, Q6_V_vror_VR(sum_sf, VLEN / 4));
    sum_sf = Q6_Vsf_vadd_VsfVsf(sum_sf, Q6_V_vror_VR(sum_sf, VLEN / 8));
    திரும்பு sum_sf;
}

நிலையான உள்வரி HVX_Vector hvx_vec_reduce_sum_f32x2(HVX_Vector in0, HVX_Vector in1) {
    HVX_VectorPair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_Vector  sum_sf  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    sum_sf = Q6_Vsf_vadd_VsfVsf(sum_sf, Q6_V_vror_VR(sum_sf, VLEN / 2));
    sum_sf = Q6_Vsf_vadd_VsfVsf(sum_sf, Q6_V_vror_VR(sum_sf, VLEN / 4));
    sum_sf = Q6_Vsf_vadd_VsfVsf(sum_sf, Q6_V_vror_VR(sum_sf, VLEN / 8));
    sum_sf = Q6_Vsf_vadd_VsfVsf(sum_sf, Q6_V_vror_VR(sum_sf, VLEN / 16));
    திரும்பு sum_sf;
}

நிலையான உள்வரி HVX_Vector hvx_vec_reduce_sum_n_f32(HVX_Vector in, unsigned முழு n) {
    unsigned முழு total = n * 4;  // total vec nbytes
    unsigned முழு width = 4;      // fp32 nbytes

    HVX_Vector sum = in, sum_t;
    போது (width < total) {
        sum_t = Q6_V_vror_VR(sum, width);       // rotate right
        sum   = Q6_Vsf_vadd_VsfVsf(sum, sum_t); // elementwise sum
        width = width << 1;
    }
    திரும்பு sum;
}

#இல்லையெனில்

நிலையான உள்வரி HVX_Vector hvx_vec_reduce_sum_f32x4(HVX_Vector_x4 in) {
    HVX_VectorPair sum_p01  = Q6_W_vshuff_VVR(in.v[1], in.v[0], 4);
    HVX_VectorPair sum_p23  = Q6_W_vshuff_VVR(in.v[3], in.v[2], 4);
    HVX_Vector     sum_qf01 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(sum_p01), Q6_V_hi_W(sum_p01));
    HVX_Vector     sum_qf23 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(sum_p23), Q6_V_hi_W(sum_p23));

    HVX_VectorPair sum_p0123 = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(sum_qf23), Q6_Vsf_equals_Vqf32(sum_qf01), 8);
    HVX_Vector     sum_qf    = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(sum_p0123), Q6_V_hi_W(sum_p0123));

    sum_qf = Q6_Vqf32_vadd_Vqf32Vsf(sum_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum_qf), VLEN / 2));
    sum_qf = Q6_Vqf32_vadd_Vqf32Vsf(sum_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum_qf), VLEN / 4));
    sum_qf = Q6_Vqf32_vadd_Vqf32Vsf(sum_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum_qf), VLEN / 8));
    திரும்பு Q6_Vsf_equals_Vqf32(sum_qf);
}

நிலையான உள்வரி HVX_Vector hvx_vec_reduce_sum_f32x2(HVX_Vector in0, HVX_Vector in1) {
    HVX_VectorPair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_Vector  sum_qf  = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    sum_qf = Q6_Vqf32_vadd_Vqf32Vsf(sum_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum_qf), VLEN / 2));
    sum_qf = Q6_Vqf32_vadd_Vqf32Vsf(sum_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum_qf), VLEN / 4));
    sum_qf = Q6_Vqf32_vadd_Vqf32Vsf(sum_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum_qf), VLEN / 8));
    sum_qf = Q6_Vqf32_vadd_Vqf32Vsf(sum_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum_qf), VLEN / 16));
    திரும்பு Q6_Vsf_equals_Vqf32(sum_qf);
}

நிலையான உள்வரி HVX_Vector hvx_vec_reduce_sum_n_f32(HVX_Vector in, unsigned முழு n) {
    unsigned முழு total = n * 4;  // total vec nbytes
    unsigned முழு width = 4;      // fp32 nbytes

    HVX_Vector sum = in, sum_t;
    போது (width < total) {
        sum_t = Q6_V_vror_VR(sum, width);                               // rotate right
        sum   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(sum, sum_t));  // elementwise sum
        width = width << 1;
    }
    திரும்பு sum;
}

#endif

நிலையான உள்வரி HVX_Vector hvx_vec_reduce_sum_f32(HVX_Vector in) {
    திரும்பு hvx_vec_reduce_sum_n_f32(in, 32);
}

நிலையான உள்வரி HVX_Vector hvx_vec_reduce_max_f16(HVX_Vector in) {
    unsigned total = 128;  // total vec nbytes
    unsigned width = 2;    // fp16 nbytes

    HVX_Vector _max = in, _max_t;
    போது (width < total) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    திரும்பு _max;
}

நிலையான உள்வரி HVX_Vector hvx_vec_reduce_max2_f16(HVX_Vector in, HVX_Vector _max) {
    unsigned total = 128;  // total vec nbytes
    unsigned width = 2;    // fp32 nbytes

    HVX_Vector _max_t;

    _max = Q6_Vhf_vmax_VhfVhf(in, _max);
    போது (width < total) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    திரும்பு _max;
}

நிலையான உள்வரி HVX_Vector hvx_vec_reduce_max_f32(HVX_Vector in) {
    unsigned total = 128;  // total vec nbytes
    unsigned width = 4;    // fp32 nbytes

    HVX_Vector _max = in, _max_t;
    போது (width < total) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    திரும்பு _max;
}

நிலையான உள்வரி HVX_Vector hvx_vec_reduce_max2_f32(HVX_Vector in, HVX_Vector _max) {
    unsigned total = 128;  // total vec nbytes
    unsigned width = 4;    // fp32 nbytes

    HVX_Vector _max_t;

    _max = Q6_Vsf_vmax_VsfVsf(in, _max);
    போது (width < total) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    திரும்பு _max;
}

#define hvx_reduce_loop_body(src_type, init_vec, pad_vec, vec_op, reduce_op, scalar_reduce) \
    செய் {                                                                                    \
        src_type * restrict vsrc = (src_type *) src;                                        \
        HVX_Vector acc = init_vec;                                                          \
                                                                                            \
        மாறிலி uint32_t elem_size = அளவு(தசம);                                           \
        மாறிலி uint32_t epv  = 128 / elem_size;                                              \
        மாறிலி uint32_t nvec = இலக்கம்_elems / epv;                                              \
        மாறிலி uint32_t nloe = இலக்கம்_elems % epv;                                              \
                                                                                            \
        uint32_t i = 0;                                                                     \
        _Pragma("unroll(4)")                                                                \
        க்கு (; i < nvec; i++) {                                                             \
            acc = vec_op(acc, vsrc[i]);                                                     \
        }                                                                                   \
        எனில் (nloe) {                                                                         \
            மாறிலி தசம * srcf = (மாறிலி தசம *) src + i * epv;                             \
            HVX_Vector in = *(HVX_UVector *) srcf;                                          \
            HVX_Vector temp = Q6_V_valign_VVR(in, pad_vec, nloe * elem_size);               \
            acc = vec_op(acc, temp);                                                        \
        }                                                                                   \
        HVX_Vector v = reduce_op(acc);                                                      \
        திரும்பு scalar_reduce(v);                                                            \
    } போது(0)

#define HVX_REDUCE_MAX_OP(acc, val) Q6_Vsf_vmax_VsfVsf(acc, val)
#define HVX_REDUCE_SUM_OP(acc, val) Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(acc), val)
#define HVX_SUM_SQ_OP(acc, val) Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(val, val))
#define HVX_REDUCE_MAX_SCALAR(v) hvx_vec_get_f32(v)
#define HVX_REDUCE_SUM_SCALAR(v) hvx_vec_get_f32(Q6_Vsf_equals_Vqf32(v))

// Max variants

நிலையான உள்வரி தசம hvx_reduce_max_f32_a(மாறிலி uint8_t * restrict src, மாறிலி முழு இலக்கம்_elems) {
    HVX_Vector init_vec = hvx_vec_splat_f32(((மாறிலி தசம *) src)[0]);
    ப்ரதிஜ்ஞா((unsigned long) src % 128 == 0);
    hvx_reduce_loop_body(HVX_Vector, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

நிலையான உள்வரி தசம hvx_reduce_max_f32_u(மாறிலி uint8_t * restrict src, மாறிலி முழு இலக்கம்_elems) {
    HVX_Vector init_vec = hvx_vec_splat_f32(((மாறிலி தசம *) src)[0]);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

நிலையான உள்வரி தசம hvx_reduce_max_f32(மாறிலி uint8_t * restrict src, மாறிலி முழு இலக்கம்_elems) {
    எனில் (hex_is_aligned((வெறுமை *) src, 128)) {
        திரும்பு hvx_reduce_max_f32_a(src, இலக்கம்_elems);
    } இல்லையெனில் {
        திரும்பு hvx_reduce_max_f32_u(src, இலக்கம்_elems);
    }
}

// Sum variants

நிலையான உள்வரி தசம hvx_reduce_sum_f32_a(மாறிலி uint8_t * restrict src, மாறிலி முழு இலக்கம்_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(0);
    ப்ரதிஜ்ஞா((unsigned long) src % 128 == 0);
    hvx_reduce_loop_body(HVX_Vector, init_vec, init_vec, HVX_REDUCE_SUM_OP, hvx_vec_reduce_sum_qf32, HVX_REDUCE_SUM_SCALAR);
}

நிலையான உள்வரி தசம hvx_reduce_sum_f32_u(மாறிலி uint8_t * restrict src, மாறிலி முழு இலக்கம்_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_SUM_OP, hvx_vec_reduce_sum_qf32, HVX_REDUCE_SUM_SCALAR);
}

நிலையான உள்வரி தசம hvx_reduce_sum_f32(மாறிலி uint8_t * restrict src, மாறிலி முழு இலக்கம்_elems) {
    எனில் (hex_is_aligned((வெறுமை *) src, 128)) {
        திரும்பு hvx_reduce_sum_f32_a(src, இலக்கம்_elems);
    } இல்லையெனில் {
        திரும்பு hvx_reduce_sum_f32_u(src, இலக்கம்_elems);
    }
}

// Sum of squares variants

நிலையான உள்வரி தசம hvx_sum_of_squares_f32_a(மாறிலி uint8_t * restrict src, மாறிலி முழு இலக்கம்_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(0);
    ப்ரதிஜ்ஞா((uintptr_t) src % 128 == 0);
    hvx_reduce_loop_body(HVX_Vector, init_vec, init_vec, HVX_SUM_SQ_OP, hvx_vec_reduce_sum_qf32, HVX_REDUCE_SUM_SCALAR);
}

நிலையான உள்வரி தசம hvx_sum_of_squares_f32_u(மாறிலி uint8_t * restrict src, மாறிலி முழு இலக்கம்_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_SUM_SQ_OP, hvx_vec_reduce_sum_qf32, HVX_REDUCE_SUM_SCALAR);
}

நிலையான உள்வரி தசம hvx_sum_of_squares_f32(மாறிலி uint8_t * restrict src, மாறிலி முழு இலக்கம்_elems) {
    எனில் (hex_is_aligned((வெறுமை *) src, 128)) {
        திரும்பு hvx_sum_of_squares_f32_a(src, இலக்கம்_elems);
    } இல்லையெனில் {
        திரும்பு hvx_sum_of_squares_f32_u(src, இலக்கம்_elems);
    }
}

// Signed 32-bit Integer Max variants

நிலையான உள்வரி HVX_Vector hvx_vec_reduce_max_n_i32(HVX_Vector in, unsigned முழு n) {
    unsigned முழு total = n * 4;  // total vec nbytes
    unsigned முழு width = 4;      // int32 nbytes

    HVX_Vector max_val = in, max_t;
    போது (width < total) {
        max_t = Q6_V_vror_VR(max_val, width);          // rotate right
        max_val = Q6_Vw_vmax_VwVw(max_t, max_val);      // elementwise signed max
        width = width << 1;
    }
    திரும்பு max_val;
}

நிலையான உள்வரி HVX_Vector hvx_vec_reduce_max_i32(HVX_Vector in) {
    திரும்பு hvx_vec_reduce_max_n_i32(in, 32);
}

நிலையான உள்வரி int32_t hvx_reduce_max_i32_a(மாறிலி uint8_t * restrict src, மாறிலி முழு இலக்கம்_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(((மாறிலி int32_t *) src)[0]);
    HVX_Vector pad_vec  = Q6_V_vsplat_R(0x80000000);
    ப்ரதிஜ்ஞா((uintptr_t) src % 128 == 0);
    hvx_reduce_loop_body(HVX_Vector, init_vec, pad_vec, Q6_Vw_vmax_VwVw, hvx_vec_reduce_max_i32, hvx_vec_get_i32);
}

நிலையான உள்வரி int32_t hvx_reduce_max_i32_u(மாறிலி uint8_t * restrict src, மாறிலி முழு இலக்கம்_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(((மாறிலி int32_t *) src)[0]);
    HVX_Vector pad_vec  = Q6_V_vsplat_R(0x80000000);
    hvx_reduce_loop_body(HVX_UVector, init_vec, pad_vec, Q6_Vw_vmax_VwVw, hvx_vec_reduce_max_i32, hvx_vec_get_i32);
}

நிலையான உள்வரி int32_t hvx_reduce_max_i32(மாறிலி uint8_t * restrict src, மாறிலி முழு இலக்கம்_elems) {
    எனில் (hex_is_aligned((வெறுமை *) src, 128)) {
        திரும்பு hvx_reduce_max_i32_a(src, இலக்கம்_elems);
    } இல்லையெனில் {
        திரும்பு hvx_reduce_max_i32_u(src, இலக்கம்_elems);
    }
}

நிலையான உள்வரி வெறுமை hvx_argmax_f32(
    மாறிலி தசம * restrict src,
    uint32_t n,
    uint32_t offset,
    தசம * out_val,
    int32_t * out_idx
) {
    எனில் (n == 0) {
        *out_val = -INFINITY;
        *out_idx = (int32_t) offset;
        திரும்பு;
    }

    எனில் (n < 32 || !hex_is_aligned((வெறுமை *) src, 128)) {
        தசம best_val = src[0];
        int32_t best_idx = (int32_t) offset;
        க்கு (uint32_t i = 1; i < n; i++) {
            எனில் (src[i] > best_val) {
                best_val = src[i];
                best_idx = (int32_t) (offset + i);
            }
        }
        *out_val = best_val;
        *out_idx = best_idx;
        திரும்பு;
    }

    நிலையான மாறிலி int32_t c_lane_idx[32] __attribute__((aligned(128))) = {
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    };

    மாறிலி HVX_Vector v_init_idx = *(மாறிலி HVX_Vector *) c_lane_idx;
    மாறிலி HVX_Vector v_step     = Q6_V_vsplat_R(32);
    HVX_Vector v_cur_idx        = Q6_Vw_vadd_VwVw(v_init_idx, Q6_V_vsplat_R((int32_t) offset));
    HVX_Vector v_max_val        = hvx_vec_splat_f32(-INFINITY);
    HVX_Vector v_max_idx        = v_cur_idx;

    மாறிலி HVX_Vector * vsrc = (மாறிலி HVX_Vector *) src;
    மாறிலி uint32_t nvec = n / 32;

    க்கு (uint32_t vi = 0; vi < nvec; vi++) {
        HVX_Vector v = vsrc[vi];
        HVX_VectorPred pred = Q6_Q_vcmp_gt_VsfVsf(v, v_max_val);
        v_max_val = Q6_V_vmux_QVV(pred, v, v_max_val);
        v_max_idx = Q6_V_vmux_QVV(pred, v_cur_idx, v_max_idx);
        v_cur_idx = Q6_Vw_vadd_VwVw(v_cur_idx, v_step);
    }

    HVX_VectorAlias u_val, u_idx;
    u_val.v = v_max_val;
    u_idx.v = v_max_idx;

    தசம best_val = u_val.fp32[0];
    int32_t best_idx = (int32_t) u_idx.w[0];
    க்கு (முழு i = 1; i < 32; i++) {
        எனில் (u_val.fp32[i] > best_val) {
            best_val = u_val.fp32[i];
            best_idx = (int32_t) u_idx.w[i];
        }
    }

    க்கு (uint32_t i = nvec * 32; i < n; i++) {
        எனில் (src[i] > best_val) {
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
