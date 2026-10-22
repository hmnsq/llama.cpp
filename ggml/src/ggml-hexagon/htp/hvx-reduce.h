#यदि_अपरिभाषितम् HVX_REDUCE_H
#परिभाषय HVX_REDUCE_H

#अन्तर्भाव <math.h>
#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <assert.h>

#अन्तर्भाव "hex-utils.h"
#अन्तर्भाव "hvx-base.h"
#अन्तर्भाव "hvx-types.h"

स्थिर अन्तःस्थ HVX_सदिशम् hvx_vec_reduce_योग_n_i32(HVX_सदिशम् अन्तर्, अचिह्नित पूर्णाङ्क n) {
    अचिह्नित पूर्णाङ्क योग = n * 4;  // total vec nbytes
    अचिह्नित पूर्णाङ्क विस्तारः = 4;      // int32

    HVX_सदिशम् योग_2 = अन्तर्, योग_t;
    यावत् (विस्तारः < योग) {
        योग_t = Q6_V_vror_VR(योग_2, विस्तारः);     // rotate right
        योग_2   = Q6_Vw_vadd_VwVw(योग_t, योग_2);  // elementwise sum
        विस्तारः = विस्तारः << 1;
    }
    प्रत्यागम योग_2;
}

स्थिर अन्तःस्थ HVX_सदिशम् hvx_vec_reduce_योग_i32(HVX_सदिशम् अन्तर्) {
    प्रत्यागम hvx_vec_reduce_योग_n_i32(अन्तर्, 32);
}

स्थिर अन्तःस्थ HVX_सदिशम् hvx_vec_reduce_योग_n_qf32(HVX_सदिशम् अन्तर्, अचिह्नित पूर्णाङ्क n) {
    अचिह्नित पूर्णाङ्क योग = n * 4;  // total vec nbytes
    अचिह्नित पूर्णाङ्क विस्तारः = 4;      // fp32 nbytes

    HVX_सदिशम् योग_2 = अन्तर्, योग_t;
    यावत् (विस्तारः < योग) {
        योग_t = Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(योग_2), विस्तारः);  // rotate right
        योग_2   = Q6_Vqf32_vadd_Vqf32Vsf(योग_2, योग_t);             // elementwise sum
        विस्तारः = विस्तारः << 1;
    }
    प्रत्यागम योग_2;
}

स्थिर अन्तःस्थ HVX_सदिशम् hvx_vec_reduce_योग_qf32(HVX_सदिशम् अन्तर्) {
    प्रत्यागम hvx_vec_reduce_योग_n_qf32(अन्तर्, 32);
}

#यदि __HVX_ARCH__ > 75

स्थिर अन्तःस्थ HVX_सदिशम् hvx_vec_reduce_योग_f32x4(HVX_सदिशम्_x4 अन्तर्) {
    HVX_सदिशम्_Pair योग_p01 = Q6_W_vshuff_VVR(अन्तर्.v[1], अन्तर्.v[0], 4);
    HVX_सदिशम्_Pair योग_p23 = Q6_W_vshuff_VVR(अन्तर्.v[3], अन्तर्.v[2], 4);
    HVX_सदिशम्  योग_sf01  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(योग_p01), Q6_V_hi_W(योग_p01));
    HVX_सदिशम्  योग_sf23  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(योग_p23), Q6_V_hi_W(योग_p23));

    HVX_सदिशम्_Pair योग_p0123 = Q6_W_vshuff_VVR(योग_sf23, योग_sf01, 8);
    HVX_सदिशम्  योग_sf       = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(योग_p0123), Q6_V_hi_W(योग_p0123));

    योग_sf = Q6_Vsf_vadd_VsfVsf(योग_sf, Q6_V_vror_VR(योग_sf, VLEN / 2));
    योग_sf = Q6_Vsf_vadd_VsfVsf(योग_sf, Q6_V_vror_VR(योग_sf, VLEN / 4));
    योग_sf = Q6_Vsf_vadd_VsfVsf(योग_sf, Q6_V_vror_VR(योग_sf, VLEN / 8));
    प्रत्यागम योग_sf;
}

स्थिर अन्तःस्थ HVX_सदिशम् hvx_vec_reduce_योग_f32x2(HVX_सदिशम् in0, HVX_सदिशम् in1) {
    HVX_सदिशम्_Pair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_सदिशम्  योग_sf  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    योग_sf = Q6_Vsf_vadd_VsfVsf(योग_sf, Q6_V_vror_VR(योग_sf, VLEN / 2));
    योग_sf = Q6_Vsf_vadd_VsfVsf(योग_sf, Q6_V_vror_VR(योग_sf, VLEN / 4));
    योग_sf = Q6_Vsf_vadd_VsfVsf(योग_sf, Q6_V_vror_VR(योग_sf, VLEN / 8));
    योग_sf = Q6_Vsf_vadd_VsfVsf(योग_sf, Q6_V_vror_VR(योग_sf, VLEN / 16));
    प्रत्यागम योग_sf;
}

स्थिर अन्तःस्थ HVX_सदिशम् hvx_vec_reduce_योग_n_f32(HVX_सदिशम् अन्तर्, अचिह्नित पूर्णाङ्क n) {
    अचिह्नित पूर्णाङ्क योग = n * 4;  // total vec nbytes
    अचिह्नित पूर्णाङ्क विस्तारः = 4;      // fp32 nbytes

    HVX_सदिशम् योग_2 = अन्तर्, योग_t;
    यावत् (विस्तारः < योग) {
        योग_t = Q6_V_vror_VR(योग_2, विस्तारः);       // rotate right
        योग_2   = Q6_Vsf_vadd_VsfVsf(योग_2, योग_t); // elementwise sum
        विस्तारः = विस्तारः << 1;
    }
    प्रत्यागम योग_2;
}

#अन्यथा

स्थिर अन्तःस्थ HVX_सदिशम् hvx_vec_reduce_योग_f32x4(HVX_सदिशम्_x4 अन्तर्) {
    HVX_सदिशम्_Pair योग_p01  = Q6_W_vshuff_VVR(अन्तर्.v[1], अन्तर्.v[0], 4);
    HVX_सदिशम्_Pair योग_p23  = Q6_W_vshuff_VVR(अन्तर्.v[3], अन्तर्.v[2], 4);
    HVX_सदिशम्     योग_qf01 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(योग_p01), Q6_V_hi_W(योग_p01));
    HVX_सदिशम्     योग_qf23 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(योग_p23), Q6_V_hi_W(योग_p23));

    HVX_सदिशम्_Pair योग_p0123 = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(योग_qf23), Q6_Vsf_equals_Vqf32(योग_qf01), 8);
    HVX_सदिशम्     योग_qf    = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(योग_p0123), Q6_V_hi_W(योग_p0123));

    योग_qf = Q6_Vqf32_vadd_Vqf32Vsf(योग_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(योग_qf), VLEN / 2));
    योग_qf = Q6_Vqf32_vadd_Vqf32Vsf(योग_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(योग_qf), VLEN / 4));
    योग_qf = Q6_Vqf32_vadd_Vqf32Vsf(योग_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(योग_qf), VLEN / 8));
    प्रत्यागम Q6_Vsf_equals_Vqf32(योग_qf);
}

स्थिर अन्तःस्थ HVX_सदिशम् hvx_vec_reduce_योग_f32x2(HVX_सदिशम् in0, HVX_सदिशम् in1) {
    HVX_सदिशम्_Pair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_सदिशम्  योग_qf  = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    योग_qf = Q6_Vqf32_vadd_Vqf32Vsf(योग_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(योग_qf), VLEN / 2));
    योग_qf = Q6_Vqf32_vadd_Vqf32Vsf(योग_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(योग_qf), VLEN / 4));
    योग_qf = Q6_Vqf32_vadd_Vqf32Vsf(योग_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(योग_qf), VLEN / 8));
    योग_qf = Q6_Vqf32_vadd_Vqf32Vsf(योग_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(योग_qf), VLEN / 16));
    प्रत्यागम Q6_Vsf_equals_Vqf32(योग_qf);
}

स्थिर अन्तःस्थ HVX_सदिशम् hvx_vec_reduce_योग_n_f32(HVX_सदिशम् अन्तर्, अचिह्नित पूर्णाङ्क n) {
    अचिह्नित पूर्णाङ्क योग = n * 4;  // total vec nbytes
    अचिह्नित पूर्णाङ्क विस्तारः = 4;      // fp32 nbytes

    HVX_सदिशम् योग_2 = अन्तर्, योग_t;
    यावत् (विस्तारः < योग) {
        योग_t = Q6_V_vror_VR(योग_2, विस्तारः);                               // rotate right
        योग_2   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(योग_2, योग_t));  // elementwise sum
        विस्तारः = विस्तारः << 1;
    }
    प्रत्यागम योग_2;
}

#यदि_अन्तः

स्थिर अन्तःस्थ HVX_सदिशम् hvx_vec_reduce_योग_f32(HVX_सदिशम् अन्तर्) {
    प्रत्यागम hvx_vec_reduce_योग_n_f32(अन्तर्, 32);
}

स्थिर अन्तःस्थ HVX_सदिशम् hvx_vec_reduce_max_f16(HVX_सदिशम् अन्तर्) {
    अचिह्नित योग = 128;  // total vec nbytes
    अचिह्नित विस्तारः = 2;    // fp16 nbytes

    HVX_सदिशम् _max = अन्तर्, _max_t;
    यावत् (विस्तारः < योग) {
        _max_t = Q6_V_vror_VR(_max, विस्तारः);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        विस्तारः  = विस्तारः << 1;
    }

    प्रत्यागम _max;
}

स्थिर अन्तःस्थ HVX_सदिशम् hvx_vec_reduce_max2_f16(HVX_सदिशम् अन्तर्, HVX_सदिशम् _max) {
    अचिह्नित योग = 128;  // total vec nbytes
    अचिह्नित विस्तारः = 2;    // fp32 nbytes

    HVX_सदिशम् _max_t;

    _max = Q6_Vhf_vmax_VhfVhf(अन्तर्, _max);
    यावत् (विस्तारः < योग) {
        _max_t = Q6_V_vror_VR(_max, विस्तारः);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        विस्तारः  = विस्तारः << 1;
    }

    प्रत्यागम _max;
}

स्थिर अन्तःस्थ HVX_सदिशम् hvx_vec_reduce_max_f32(HVX_सदिशम् अन्तर्) {
    अचिह्नित योग = 128;  // total vec nbytes
    अचिह्नित विस्तारः = 4;    // fp32 nbytes

    HVX_सदिशम् _max = अन्तर्, _max_t;
    यावत् (विस्तारः < योग) {
        _max_t = Q6_V_vror_VR(_max, विस्तारः);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        विस्तारः  = विस्तारः << 1;
    }

    प्रत्यागम _max;
}

स्थिर अन्तःस्थ HVX_सदिशम् hvx_vec_reduce_max2_f32(HVX_सदिशम् अन्तर्, HVX_सदिशम् _max) {
    अचिह्नित योग = 128;  // total vec nbytes
    अचिह्नित विस्तारः = 4;    // fp32 nbytes

    HVX_सदिशम् _max_t;

    _max = Q6_Vsf_vmax_VsfVsf(अन्तर्, _max);
    यावत् (विस्तारः < योग) {
        _max_t = Q6_V_vror_VR(_max, विस्तारः);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        विस्तारः  = विस्तारः << 1;
    }

    प्रत्यागम _max;
}

#परिभाषय hvx_reduce_loop_body(src_type, init_vec, pad_vec, vec_op, reduce_op, scalar_reduce) \
    कुरु {                                                                                    \
        src_type * सीमित vsrc = (src_type *) src;                                        \
        HVX_सदिशम् acc = init_vec;                                                          \
                                                                                            \
        ध्रुव uint32_t elem_आकार = परिमाण(भिन्न);                                           \
        ध्रुव uint32_t epv  = 128 / elem_आकार;                                              \
        ध्रुव uint32_t nvec = अङ्कः_elems / epv;                                              \
        ध्रुव uint32_t nloe = अङ्कः_elems % epv;                                              \
                                                                                            \
        uint32_t i = 0;                                                                     \
        _Pragma("unroll(4)")                                                                \
        प्रति (; i < nvec; i++) {                                                             \
            acc = vec_op(acc, vsrc[i]);                                                     \
        }                                                                                   \
        यदि (nloe) {                                                                         \
            ध्रुव भिन्न * srcf = (ध्रुव भिन्न *) src + i * epv;                             \
            HVX_सदिशम् अन्तर् = *(HVX_UVector *) srcf;                                          \
            HVX_सदिशम् अस्थायी = Q6_V_valign_VVR(अन्तर्, pad_vec, nloe * elem_आकार);               \
            acc = vec_op(acc, अस्थायी);                                                        \
        }                                                                                   \
        HVX_सदिशम् v = reduce_op(acc);                                                      \
        प्रत्यागम scalar_reduce(v);                                                            \
    } यावत्(0)

#परिभाषय HVX_REDUCE_MAX_OP(acc, val) Q6_Vsf_vmax_VsfVsf(acc, val)
#परिभाषय HVX_REDUCE_योग_OP(acc, val) Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(acc), val)
#परिभाषय HVX_योग_SQ_OP(acc, val) Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(val, val))
#परिभाषय HVX_REDUCE_MAX_SCALAR(v) hvx_vec_get_f32(v)
#परिभाषय HVX_REDUCE_योग_SCALAR(v) hvx_vec_get_f32(Q6_Vsf_equals_Vqf32(v))

// Max variants

स्थिर अन्तःस्थ भिन्न hvx_reduce_max_f32_a(ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क अङ्कः_elems) {
    HVX_सदिशम् init_vec = hvx_vec_splat_f32(((ध्रुव भिन्न *) src)[0]);
    प्रतिज्ञा((अचिह्नित दीर्घ) src % 128 == 0);
    hvx_reduce_loop_body(HVX_सदिशम्, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

स्थिर अन्तःस्थ भिन्न hvx_reduce_max_f32_u(ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क अङ्कः_elems) {
    HVX_सदिशम् init_vec = hvx_vec_splat_f32(((ध्रुव भिन्न *) src)[0]);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

स्थिर अन्तःस्थ भिन्न hvx_reduce_max_f32(ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क अङ्कः_elems) {
    यदि (hex_is_aligned((शून्य *) src, 128)) {
        प्रत्यागम hvx_reduce_max_f32_a(src, अङ्कः_elems);
    } अन्यथा {
        प्रत्यागम hvx_reduce_max_f32_u(src, अङ्कः_elems);
    }
}

// Sum variants

स्थिर अन्तःस्थ भिन्न hvx_reduce_योग_f32_a(ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क अङ्कः_elems) {
    HVX_सदिशम् init_vec = Q6_V_vsplat_R(0);
    प्रतिज्ञा((अचिह्नित दीर्घ) src % 128 == 0);
    hvx_reduce_loop_body(HVX_सदिशम्, init_vec, init_vec, HVX_REDUCE_योग_OP, hvx_vec_reduce_योग_qf32, HVX_REDUCE_योग_SCALAR);
}

स्थिर अन्तःस्थ भिन्न hvx_reduce_योग_f32_u(ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क अङ्कः_elems) {
    HVX_सदिशम् init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_योग_OP, hvx_vec_reduce_योग_qf32, HVX_REDUCE_योग_SCALAR);
}

स्थिर अन्तःस्थ भिन्न hvx_reduce_योग_f32(ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क अङ्कः_elems) {
    यदि (hex_is_aligned((शून्य *) src, 128)) {
        प्रत्यागम hvx_reduce_योग_f32_a(src, अङ्कः_elems);
    } अन्यथा {
        प्रत्यागम hvx_reduce_योग_f32_u(src, अङ्कः_elems);
    }
}

// Sum of squares variants

स्थिर अन्तःस्थ भिन्न hvx_योग_सम्बन्धि_squares_f32_a(ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क अङ्कः_elems) {
    HVX_सदिशम् init_vec = Q6_V_vsplat_R(0);
    प्रतिज्ञा((uintptr_t) src % 128 == 0);
    hvx_reduce_loop_body(HVX_सदिशम्, init_vec, init_vec, HVX_योग_SQ_OP, hvx_vec_reduce_योग_qf32, HVX_REDUCE_योग_SCALAR);
}

स्थिर अन्तःस्थ भिन्न hvx_योग_सम्बन्धि_squares_f32_u(ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क अङ्कः_elems) {
    HVX_सदिशम् init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_योग_SQ_OP, hvx_vec_reduce_योग_qf32, HVX_REDUCE_योग_SCALAR);
}

स्थिर अन्तःस्थ भिन्न hvx_योग_सम्बन्धि_squares_f32(ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क अङ्कः_elems) {
    यदि (hex_is_aligned((शून्य *) src, 128)) {
        प्रत्यागम hvx_योग_सम्बन्धि_squares_f32_a(src, अङ्कः_elems);
    } अन्यथा {
        प्रत्यागम hvx_योग_सम्बन्धि_squares_f32_u(src, अङ्कः_elems);
    }
}

// Signed 32-bit Integer Max variants

स्थिर अन्तःस्थ HVX_सदिशम् hvx_vec_reduce_max_n_i32(HVX_सदिशम् अन्तर्, अचिह्नित पूर्णाङ्क n) {
    अचिह्नित पूर्णाङ्क योग = n * 4;  // total vec nbytes
    अचिह्नित पूर्णाङ्क विस्तारः = 4;      // int32 nbytes

    HVX_सदिशम् max_val = अन्तर्, max_t;
    यावत् (विस्तारः < योग) {
        max_t = Q6_V_vror_VR(max_val, विस्तारः);          // rotate right
        max_val = Q6_Vw_vmax_VwVw(max_t, max_val);      // elementwise signed max
        विस्तारः = विस्तारः << 1;
    }
    प्रत्यागम max_val;
}

स्थिर अन्तःस्थ HVX_सदिशम् hvx_vec_reduce_max_i32(HVX_सदिशम् अन्तर्) {
    प्रत्यागम hvx_vec_reduce_max_n_i32(अन्तर्, 32);
}

स्थिर अन्तःस्थ int32_t hvx_reduce_max_i32_a(ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क अङ्कः_elems) {
    HVX_सदिशम् init_vec = Q6_V_vsplat_R(((ध्रुव int32_t *) src)[0]);
    HVX_सदिशम् pad_vec  = Q6_V_vsplat_R(0x80000000);
    प्रतिज्ञा((uintptr_t) src % 128 == 0);
    hvx_reduce_loop_body(HVX_सदिशम्, init_vec, pad_vec, Q6_Vw_vmax_VwVw, hvx_vec_reduce_max_i32, hvx_vec_get_i32);
}

स्थिर अन्तःस्थ int32_t hvx_reduce_max_i32_u(ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क अङ्कः_elems) {
    HVX_सदिशम् init_vec = Q6_V_vsplat_R(((ध्रुव int32_t *) src)[0]);
    HVX_सदिशम् pad_vec  = Q6_V_vsplat_R(0x80000000);
    hvx_reduce_loop_body(HVX_UVector, init_vec, pad_vec, Q6_Vw_vmax_VwVw, hvx_vec_reduce_max_i32, hvx_vec_get_i32);
}

स्थिर अन्तःस्थ int32_t hvx_reduce_max_i32(ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क अङ्कः_elems) {
    यदि (hex_is_aligned((शून्य *) src, 128)) {
        प्रत्यागम hvx_reduce_max_i32_a(src, अङ्कः_elems);
    } अन्यथा {
        प्रत्यागम hvx_reduce_max_i32_u(src, अङ्कः_elems);
    }
}

स्थिर अन्तःस्थ शून्य hvx_argmax_f32(
    ध्रुव भिन्न * सीमित src,
    uint32_t n,
    uint32_t विस्थापनम्,
    भिन्न * बहिर्गमनम्_val,
    int32_t * बहिर्गमनम्_idx
) {
    यदि (n == 0) {
        *बहिर्गमनम्_val = -INFINITY;
        *बहिर्गमनम्_idx = (int32_t) विस्थापनम्;
        प्रत्यागम;
    }

    यदि (n < 32 || !hex_is_aligned((शून्य *) src, 128)) {
        भिन्न best_val = src[0];
        int32_t best_idx = (int32_t) विस्थापनम्;
        प्रति (uint32_t i = 1; i < n; i++) {
            यदि (src[i] > best_val) {
                best_val = src[i];
                best_idx = (int32_t) (विस्थापनम् + i);
            }
        }
        *बहिर्गमनम्_val = best_val;
        *बहिर्गमनम्_idx = best_idx;
        प्रत्यागम;
    }

    स्थिर ध्रुव int32_t c_lane_idx[32] __attribute__((aligned(128))) = {
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    };

    ध्रुव HVX_सदिशम् v_init_idx = *(ध्रुव HVX_सदिशम् *) c_lane_idx;
    ध्रुव HVX_सदिशम् v_चरणः     = Q6_V_vsplat_R(32);
    HVX_सदिशम् v_cur_idx        = Q6_Vw_vadd_VwVw(v_init_idx, Q6_V_vsplat_R((int32_t) विस्थापनम्));
    HVX_सदिशम् v_max_val        = hvx_vec_splat_f32(-INFINITY);
    HVX_सदिशम् v_max_idx        = v_cur_idx;

    ध्रुव HVX_सदिशम् * vsrc = (ध्रुव HVX_सदिशम् *) src;
    ध्रुव uint32_t nvec = n / 32;

    प्रति (uint32_t vi = 0; vi < nvec; vi++) {
        HVX_सदिशम् v = vsrc[vi];
        HVX_सदिशम्_Pred pred = Q6_Q_vcmp_gt_VsfVsf(v, v_max_val);
        v_max_val = Q6_V_vmux_QVV(pred, v, v_max_val);
        v_max_idx = Q6_V_vmux_QVV(pred, v_cur_idx, v_max_idx);
        v_cur_idx = Q6_Vw_vadd_VwVw(v_cur_idx, v_चरणः);
    }

    HVX_सदिशम्_Alias u_val, u_idx;
    u_val.v = v_max_val;
    u_idx.v = v_max_idx;

    भिन्न best_val = u_val.fp32[0];
    int32_t best_idx = (int32_t) u_idx.w[0];
    प्रति (पूर्णाङ्क i = 1; i < 32; i++) {
        यदि (u_val.fp32[i] > best_val) {
            best_val = u_val.fp32[i];
            best_idx = (int32_t) u_idx.w[i];
        }
    }

    प्रति (uint32_t i = nvec * 32; i < n; i++) {
        यदि (src[i] > best_val) {
            best_val = src[i];
            best_idx = (int32_t) (विस्थापनम् + i);
        }
    }

    *बहिर्गमनम्_val = best_val;
    *बहिर्गमनम्_idx = best_idx;
}

#अपरिभाषय hvx_reduce_loop_body
#अपरिभाषय HVX_REDUCE_MAX_OP
#अपरिभाषय HVX_REDUCE_योग_OP
#अपरिभाषय HVX_REDUCE_MAX_SCALAR
#अपरिभाषय HVX_REDUCE_योग_SCALAR
#अपरिभाषय HVX_योग_SQ_OP

#यदि_अन्तः /* HVX_REDUCE_H */
