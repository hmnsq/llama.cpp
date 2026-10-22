#अगर_अघोषित HVX_REDUCE_H
#घोषणा HVX_REDUCE_H

#समावेश <math.h>
#समावेश <stdbool.h>
#समावेश <stdint.h>
#समावेश <assert.h>

#समावेश "hex-utils.h"
#समावेश "hvx-base.h"
#समावेश "hvx-types.h"

अटल अंतरभूत HVX_Vector hvx_vec_reduce_योग_n_i32(HVX_Vector in, अचिन्हित पूर्णांक n) {
    अचिन्हित पूर्णांक योग = n * 4;  // total vec nbytes
    अचिन्हित पूर्णांक width = 4;      // int32

    HVX_Vector योग_2 = in, योग_t;
    जबतक (width < योग) {
        योग_t = Q6_V_vror_VR(योग_2, width);     // rotate right
        योग_2   = Q6_Vw_vadd_VwVw(योग_t, योग_2);  // elementwise sum
        width = width << 1;
    }
    वापस योग_2;
}

अटल अंतरभूत HVX_Vector hvx_vec_reduce_योग_i32(HVX_Vector in) {
    वापस hvx_vec_reduce_योग_n_i32(in, 32);
}

अटल अंतरभूत HVX_Vector hvx_vec_reduce_योग_n_qf32(HVX_Vector in, अचिन्हित पूर्णांक n) {
    अचिन्हित पूर्णांक योग = n * 4;  // total vec nbytes
    अचिन्हित पूर्णांक width = 4;      // fp32 nbytes

    HVX_Vector योग_2 = in, योग_t;
    जबतक (width < योग) {
        योग_t = Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(योग_2), width);  // rotate right
        योग_2   = Q6_Vqf32_vadd_Vqf32Vsf(योग_2, योग_t);             // elementwise sum
        width = width << 1;
    }
    वापस योग_2;
}

अटल अंतरभूत HVX_Vector hvx_vec_reduce_योग_qf32(HVX_Vector in) {
    वापस hvx_vec_reduce_योग_n_qf32(in, 32);
}

#अगर __HVX_ARCH__ > 75

अटल अंतरभूत HVX_Vector hvx_vec_reduce_योग_f32x4(HVX_Vector_x4 in) {
    HVX_VectorPair योग_p01 = Q6_W_vshuff_VVR(in.v[1], in.v[0], 4);
    HVX_VectorPair योग_p23 = Q6_W_vshuff_VVR(in.v[3], in.v[2], 4);
    HVX_Vector  योग_sf01  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(योग_p01), Q6_V_hi_W(योग_p01));
    HVX_Vector  योग_sf23  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(योग_p23), Q6_V_hi_W(योग_p23));

    HVX_VectorPair योग_p0123 = Q6_W_vshuff_VVR(योग_sf23, योग_sf01, 8);
    HVX_Vector  योग_sf       = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(योग_p0123), Q6_V_hi_W(योग_p0123));

    योग_sf = Q6_Vsf_vadd_VsfVsf(योग_sf, Q6_V_vror_VR(योग_sf, VLEN / 2));
    योग_sf = Q6_Vsf_vadd_VsfVsf(योग_sf, Q6_V_vror_VR(योग_sf, VLEN / 4));
    योग_sf = Q6_Vsf_vadd_VsfVsf(योग_sf, Q6_V_vror_VR(योग_sf, VLEN / 8));
    वापस योग_sf;
}

अटल अंतरभूत HVX_Vector hvx_vec_reduce_योग_f32x2(HVX_Vector in0, HVX_Vector in1) {
    HVX_VectorPair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_Vector  योग_sf  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    योग_sf = Q6_Vsf_vadd_VsfVsf(योग_sf, Q6_V_vror_VR(योग_sf, VLEN / 2));
    योग_sf = Q6_Vsf_vadd_VsfVsf(योग_sf, Q6_V_vror_VR(योग_sf, VLEN / 4));
    योग_sf = Q6_Vsf_vadd_VsfVsf(योग_sf, Q6_V_vror_VR(योग_sf, VLEN / 8));
    योग_sf = Q6_Vsf_vadd_VsfVsf(योग_sf, Q6_V_vror_VR(योग_sf, VLEN / 16));
    वापस योग_sf;
}

अटल अंतरभूत HVX_Vector hvx_vec_reduce_योग_n_f32(HVX_Vector in, अचिन्हित पूर्णांक n) {
    अचिन्हित पूर्णांक योग = n * 4;  // total vec nbytes
    अचिन्हित पूर्णांक width = 4;      // fp32 nbytes

    HVX_Vector योग_2 = in, योग_t;
    जबतक (width < योग) {
        योग_t = Q6_V_vror_VR(योग_2, width);       // rotate right
        योग_2   = Q6_Vsf_vadd_VsfVsf(योग_2, योग_t); // elementwise sum
        width = width << 1;
    }
    वापस योग_2;
}

#अन्यथा

अटल अंतरभूत HVX_Vector hvx_vec_reduce_योग_f32x4(HVX_Vector_x4 in) {
    HVX_VectorPair योग_p01  = Q6_W_vshuff_VVR(in.v[1], in.v[0], 4);
    HVX_VectorPair योग_p23  = Q6_W_vshuff_VVR(in.v[3], in.v[2], 4);
    HVX_Vector     योग_qf01 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(योग_p01), Q6_V_hi_W(योग_p01));
    HVX_Vector     योग_qf23 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(योग_p23), Q6_V_hi_W(योग_p23));

    HVX_VectorPair योग_p0123 = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(योग_qf23), Q6_Vsf_equals_Vqf32(योग_qf01), 8);
    HVX_Vector     योग_qf    = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(योग_p0123), Q6_V_hi_W(योग_p0123));

    योग_qf = Q6_Vqf32_vadd_Vqf32Vsf(योग_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(योग_qf), VLEN / 2));
    योग_qf = Q6_Vqf32_vadd_Vqf32Vsf(योग_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(योग_qf), VLEN / 4));
    योग_qf = Q6_Vqf32_vadd_Vqf32Vsf(योग_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(योग_qf), VLEN / 8));
    वापस Q6_Vsf_equals_Vqf32(योग_qf);
}

अटल अंतरभूत HVX_Vector hvx_vec_reduce_योग_f32x2(HVX_Vector in0, HVX_Vector in1) {
    HVX_VectorPair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_Vector  योग_qf  = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    योग_qf = Q6_Vqf32_vadd_Vqf32Vsf(योग_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(योग_qf), VLEN / 2));
    योग_qf = Q6_Vqf32_vadd_Vqf32Vsf(योग_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(योग_qf), VLEN / 4));
    योग_qf = Q6_Vqf32_vadd_Vqf32Vsf(योग_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(योग_qf), VLEN / 8));
    योग_qf = Q6_Vqf32_vadd_Vqf32Vsf(योग_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(योग_qf), VLEN / 16));
    वापस Q6_Vsf_equals_Vqf32(योग_qf);
}

अटल अंतरभूत HVX_Vector hvx_vec_reduce_योग_n_f32(HVX_Vector in, अचिन्हित पूर्णांक n) {
    अचिन्हित पूर्णांक योग = n * 4;  // total vec nbytes
    अचिन्हित पूर्णांक width = 4;      // fp32 nbytes

    HVX_Vector योग_2 = in, योग_t;
    जबतक (width < योग) {
        योग_t = Q6_V_vror_VR(योग_2, width);                               // rotate right
        योग_2   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(योग_2, योग_t));  // elementwise sum
        width = width << 1;
    }
    वापस योग_2;
}

#पूर्ण_अगर

अटल अंतरभूत HVX_Vector hvx_vec_reduce_योग_f32(HVX_Vector in) {
    वापस hvx_vec_reduce_योग_n_f32(in, 32);
}

अटल अंतरभूत HVX_Vector hvx_vec_reduce_max_f16(HVX_Vector in) {
    अचिन्हित योग = 128;  // total vec nbytes
    अचिन्हित width = 2;    // fp16 nbytes

    HVX_Vector _max = in, _max_t;
    जबतक (width < योग) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    वापस _max;
}

अटल अंतरभूत HVX_Vector hvx_vec_reduce_max2_f16(HVX_Vector in, HVX_Vector _max) {
    अचिन्हित योग = 128;  // total vec nbytes
    अचिन्हित width = 2;    // fp32 nbytes

    HVX_Vector _max_t;

    _max = Q6_Vhf_vmax_VhfVhf(in, _max);
    जबतक (width < योग) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    वापस _max;
}

अटल अंतरभूत HVX_Vector hvx_vec_reduce_max_f32(HVX_Vector in) {
    अचिन्हित योग = 128;  // total vec nbytes
    अचिन्हित width = 4;    // fp32 nbytes

    HVX_Vector _max = in, _max_t;
    जबतक (width < योग) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    वापस _max;
}

अटल अंतरभूत HVX_Vector hvx_vec_reduce_max2_f32(HVX_Vector in, HVX_Vector _max) {
    अचिन्हित योग = 128;  // total vec nbytes
    अचिन्हित width = 4;    // fp32 nbytes

    HVX_Vector _max_t;

    _max = Q6_Vsf_vmax_VsfVsf(in, _max);
    जबतक (width < योग) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    वापस _max;
}

#घोषणा hvx_reduce_loop_body(src_type, init_vec, pad_vec, vec_op, reduce_op, scalar_reduce) \
    करो {                                                                                    \
        src_type * restrict vsrc = (src_type *) src;                                        \
        HVX_Vector acc = init_vec;                                                          \
                                                                                            \
        स्थिर uint32_t elem_आकार = माप(भग्न);                                           \
        स्थिर uint32_t epv  = 128 / elem_आकार;                                              \
        स्थिर uint32_t nvec = अंक_elems / epv;                                              \
        स्थिर uint32_t nloe = अंक_elems % epv;                                              \
                                                                                            \
        uint32_t i = 0;                                                                     \
        _Pragma("unroll(4)")                                                                \
        क्रम (; i < nvec; i++) {                                                             \
            acc = vec_op(acc, vsrc[i]);                                                     \
        }                                                                                   \
        अगर (nloe) {                                                                         \
            स्थिर भग्न * srcf = (स्थिर भग्न *) src + i * epv;                             \
            HVX_Vector in = *(HVX_UVector *) srcf;                                          \
            HVX_Vector अस्थायी = Q6_V_valign_VVR(in, pad_vec, nloe * elem_आकार);               \
            acc = vec_op(acc, अस्थायी);                                                        \
        }                                                                                   \
        HVX_Vector v = reduce_op(acc);                                                      \
        वापस scalar_reduce(v);                                                            \
    } जबतक(0)

#घोषणा HVX_REDUCE_MAX_OP(acc, val) Q6_Vsf_vmax_VsfVsf(acc, val)
#घोषणा HVX_REDUCE_योग_OP(acc, val) Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(acc), val)
#घोषणा HVX_योग_SQ_OP(acc, val) Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(val, val))
#घोषणा HVX_REDUCE_MAX_SCALAR(v) hvx_vec_get_f32(v)
#घोषणा HVX_REDUCE_योग_SCALAR(v) hvx_vec_get_f32(Q6_Vsf_equals_Vqf32(v))

// Max variants

अटल अंतरभूत भग्न hvx_reduce_max_f32_a(स्थिर uint8_t * restrict src, स्थिर पूर्णांक अंक_elems) {
    HVX_Vector init_vec = hvx_vec_splat_f32(((स्थिर भग्न *) src)[0]);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_reduce_loop_body(HVX_Vector, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

अटल अंतरभूत भग्न hvx_reduce_max_f32_u(स्थिर uint8_t * restrict src, स्थिर पूर्णांक अंक_elems) {
    HVX_Vector init_vec = hvx_vec_splat_f32(((स्थिर भग्न *) src)[0]);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

अटल अंतरभूत भग्न hvx_reduce_max_f32(स्थिर uint8_t * restrict src, स्थिर पूर्णांक अंक_elems) {
    अगर (hex_is_aligned((व्योम *) src, 128)) {
        वापस hvx_reduce_max_f32_a(src, अंक_elems);
    } अन्यथा {
        वापस hvx_reduce_max_f32_u(src, अंक_elems);
    }
}

// Sum variants

अटल अंतरभूत भग्न hvx_reduce_योग_f32_a(स्थिर uint8_t * restrict src, स्थिर पूर्णांक अंक_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(0);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_reduce_loop_body(HVX_Vector, init_vec, init_vec, HVX_REDUCE_योग_OP, hvx_vec_reduce_योग_qf32, HVX_REDUCE_योग_SCALAR);
}

अटल अंतरभूत भग्न hvx_reduce_योग_f32_u(स्थिर uint8_t * restrict src, स्थिर पूर्णांक अंक_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_योग_OP, hvx_vec_reduce_योग_qf32, HVX_REDUCE_योग_SCALAR);
}

अटल अंतरभूत भग्न hvx_reduce_योग_f32(स्थिर uint8_t * restrict src, स्थिर पूर्णांक अंक_elems) {
    अगर (hex_is_aligned((व्योम *) src, 128)) {
        वापस hvx_reduce_योग_f32_a(src, अंक_elems);
    } अन्यथा {
        वापस hvx_reduce_योग_f32_u(src, अंक_elems);
    }
}

// Sum of squares variants

अटल अंतरभूत भग्न hvx_योग_of_squares_f32_a(स्थिर uint8_t * restrict src, स्थिर पूर्णांक अंक_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(0);
    निश्चित((uintptr_t) src % 128 == 0);
    hvx_reduce_loop_body(HVX_Vector, init_vec, init_vec, HVX_योग_SQ_OP, hvx_vec_reduce_योग_qf32, HVX_REDUCE_योग_SCALAR);
}

अटल अंतरभूत भग्न hvx_योग_of_squares_f32_u(स्थिर uint8_t * restrict src, स्थिर पूर्णांक अंक_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_योग_SQ_OP, hvx_vec_reduce_योग_qf32, HVX_REDUCE_योग_SCALAR);
}

अटल अंतरभूत भग्न hvx_योग_of_squares_f32(स्थिर uint8_t * restrict src, स्थिर पूर्णांक अंक_elems) {
    अगर (hex_is_aligned((व्योम *) src, 128)) {
        वापस hvx_योग_of_squares_f32_a(src, अंक_elems);
    } अन्यथा {
        वापस hvx_योग_of_squares_f32_u(src, अंक_elems);
    }
}

// Signed 32-bit Integer Max variants

अटल अंतरभूत HVX_Vector hvx_vec_reduce_max_n_i32(HVX_Vector in, अचिन्हित पूर्णांक n) {
    अचिन्हित पूर्णांक योग = n * 4;  // total vec nbytes
    अचिन्हित पूर्णांक width = 4;      // int32 nbytes

    HVX_Vector max_val = in, max_t;
    जबतक (width < योग) {
        max_t = Q6_V_vror_VR(max_val, width);          // rotate right
        max_val = Q6_Vw_vmax_VwVw(max_t, max_val);      // elementwise signed max
        width = width << 1;
    }
    वापस max_val;
}

अटल अंतरभूत HVX_Vector hvx_vec_reduce_max_i32(HVX_Vector in) {
    वापस hvx_vec_reduce_max_n_i32(in, 32);
}

अटल अंतरभूत int32_t hvx_reduce_max_i32_a(स्थिर uint8_t * restrict src, स्थिर पूर्णांक अंक_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(((स्थिर int32_t *) src)[0]);
    HVX_Vector pad_vec  = Q6_V_vsplat_R(0x80000000);
    निश्चित((uintptr_t) src % 128 == 0);
    hvx_reduce_loop_body(HVX_Vector, init_vec, pad_vec, Q6_Vw_vmax_VwVw, hvx_vec_reduce_max_i32, hvx_vec_get_i32);
}

अटल अंतरभूत int32_t hvx_reduce_max_i32_u(स्थिर uint8_t * restrict src, स्थिर पूर्णांक अंक_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(((स्थिर int32_t *) src)[0]);
    HVX_Vector pad_vec  = Q6_V_vsplat_R(0x80000000);
    hvx_reduce_loop_body(HVX_UVector, init_vec, pad_vec, Q6_Vw_vmax_VwVw, hvx_vec_reduce_max_i32, hvx_vec_get_i32);
}

अटल अंतरभूत int32_t hvx_reduce_max_i32(स्थिर uint8_t * restrict src, स्थिर पूर्णांक अंक_elems) {
    अगर (hex_is_aligned((व्योम *) src, 128)) {
        वापस hvx_reduce_max_i32_a(src, अंक_elems);
    } अन्यथा {
        वापस hvx_reduce_max_i32_u(src, अंक_elems);
    }
}

अटल अंतरभूत व्योम hvx_argmax_f32(
    स्थिर भग्न * restrict src,
    uint32_t n,
    uint32_t offset,
    भग्न * out_val,
    int32_t * out_idx
) {
    अगर (n == 0) {
        *out_val = -अनन्त;
        *out_idx = (int32_t) offset;
        वापस;
    }

    अगर (n < 32 || !hex_is_aligned((व्योम *) src, 128)) {
        भग्न best_val = src[0];
        int32_t best_idx = (int32_t) offset;
        क्रम (uint32_t i = 1; i < n; i++) {
            अगर (src[i] > best_val) {
                best_val = src[i];
                best_idx = (int32_t) (offset + i);
            }
        }
        *out_val = best_val;
        *out_idx = best_idx;
        वापस;
    }

    अटल स्थिर int32_t c_lane_idx[32] __attribute__((aligned(128))) = {
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    };

    स्थिर HVX_Vector v_init_idx = *(स्थिर HVX_Vector *) c_lane_idx;
    स्थिर HVX_Vector v_step     = Q6_V_vsplat_R(32);
    HVX_Vector v_cur_idx        = Q6_Vw_vadd_VwVw(v_init_idx, Q6_V_vsplat_R((int32_t) offset));
    HVX_Vector v_max_val        = hvx_vec_splat_f32(-अनन्त);
    HVX_Vector v_max_idx        = v_cur_idx;

    स्थिर HVX_Vector * vsrc = (स्थिर HVX_Vector *) src;
    स्थिर uint32_t nvec = n / 32;

    क्रम (uint32_t vi = 0; vi < nvec; vi++) {
        HVX_Vector v = vsrc[vi];
        HVX_VectorPred pred = Q6_Q_vcmp_gt_VsfVsf(v, v_max_val);
        v_max_val = Q6_V_vmux_QVV(pred, v, v_max_val);
        v_max_idx = Q6_V_vmux_QVV(pred, v_cur_idx, v_max_idx);
        v_cur_idx = Q6_Vw_vadd_VwVw(v_cur_idx, v_step);
    }

    HVX_VectorAlias u_val, u_idx;
    u_val.v = v_max_val;
    u_idx.v = v_max_idx;

    भग्न best_val = u_val.fp32[0];
    int32_t best_idx = (int32_t) u_idx.w[0];
    क्रम (पूर्णांक i = 1; i < 32; i++) {
        अगर (u_val.fp32[i] > best_val) {
            best_val = u_val.fp32[i];
            best_idx = (int32_t) u_idx.w[i];
        }
    }

    क्रम (uint32_t i = nvec * 32; i < n; i++) {
        अगर (src[i] > best_val) {
            best_val = src[i];
            best_idx = (int32_t) (offset + i);
        }
    }

    *out_val = best_val;
    *out_idx = best_idx;
}

#अघोषित hvx_reduce_loop_body
#अघोषित HVX_REDUCE_MAX_OP
#अघोषित HVX_REDUCE_योग_OP
#अघोषित HVX_REDUCE_MAX_SCALAR
#अघोषित HVX_REDUCE_योग_SCALAR
#अघोषित HVX_योग_SQ_OP

#पूर्ण_अगर /* HVX_REDUCE_H */
