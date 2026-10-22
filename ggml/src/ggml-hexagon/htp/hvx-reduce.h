#جيڪڏهن_وصف_نه_ٿيل HVX_REDUCE_H
#وصف_ڪريو HVX_REDUCE_H

#ضمن <math.h>
#ضمن <stdbool.h>
#ضمن <stdint.h>
#ضمن <assert.h>

#ضمن "hex-utils.h"
#ضمن "hvx-base.h"
#ضمن "hvx-types.h"

ساكن مضمن HVX_ويڪٽر hvx_vec_reduce_جوڙ_n_i32(HVX_ويڪٽر اندريون, بنانشان صحيح n) {
    بنانشان صحيح ڪل = n * 4;  // total vec nbytes
    بنانشان صحيح ويڪر = 4;      // int32

    HVX_ويڪٽر جوڙ = اندريون, جوڙ_t;
    طالما (ويڪر < ڪل) {
        جوڙ_t = Q6_V_vror_VR(جوڙ, ويڪر);     // rotate right
        جوڙ   = Q6_Vw_vadd_VwVw(جوڙ_t, جوڙ);  // elementwise sum
        ويڪر = ويڪر << 1;
    }
    أرجع جوڙ;
}

ساكن مضمن HVX_ويڪٽر hvx_vec_reduce_جوڙ_i32(HVX_ويڪٽر اندريون) {
    أرجع hvx_vec_reduce_جوڙ_n_i32(اندريون, 32);
}

ساكن مضمن HVX_ويڪٽر hvx_vec_reduce_جوڙ_n_qf32(HVX_ويڪٽر اندريون, بنانشان صحيح n) {
    بنانشان صحيح ڪل = n * 4;  // total vec nbytes
    بنانشان صحيح ويڪر = 4;      // fp32 nbytes

    HVX_ويڪٽر جوڙ = اندريون, جوڙ_t;
    طالما (ويڪر < ڪل) {
        جوڙ_t = Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(جوڙ), ويڪر);  // rotate right
        جوڙ   = Q6_Vqf32_vadd_Vqf32Vsf(جوڙ, جوڙ_t);             // elementwise sum
        ويڪر = ويڪر << 1;
    }
    أرجع جوڙ;
}

ساكن مضمن HVX_ويڪٽر hvx_vec_reduce_جوڙ_qf32(HVX_ويڪٽر اندريون) {
    أرجع hvx_vec_reduce_جوڙ_n_qf32(اندريون, 32);
}

#إذا __HVX_ARCH__ > 75

ساكن مضمن HVX_ويڪٽر hvx_vec_reduce_جوڙ_f32x4(HVX_ويڪٽر_x4 اندريون) {
    HVX_ويڪٽر_Pair جوڙ_p01 = Q6_W_vshuff_VVR(اندريون.v[1], اندريون.v[0], 4);
    HVX_ويڪٽر_Pair جوڙ_p23 = Q6_W_vshuff_VVR(اندريون.v[3], اندريون.v[2], 4);
    HVX_ويڪٽر  جوڙ_sf01  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(جوڙ_p01), Q6_V_hi_W(جوڙ_p01));
    HVX_ويڪٽر  جوڙ_sf23  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(جوڙ_p23), Q6_V_hi_W(جوڙ_p23));

    HVX_ويڪٽر_Pair جوڙ_p0123 = Q6_W_vshuff_VVR(جوڙ_sf23, جوڙ_sf01, 8);
    HVX_ويڪٽر  جوڙ_sf       = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(جوڙ_p0123), Q6_V_hi_W(جوڙ_p0123));

    جوڙ_sf = Q6_Vsf_vadd_VsfVsf(جوڙ_sf, Q6_V_vror_VR(جوڙ_sf, VLEN / 2));
    جوڙ_sf = Q6_Vsf_vadd_VsfVsf(جوڙ_sf, Q6_V_vror_VR(جوڙ_sf, VLEN / 4));
    جوڙ_sf = Q6_Vsf_vadd_VsfVsf(جوڙ_sf, Q6_V_vror_VR(جوڙ_sf, VLEN / 8));
    أرجع جوڙ_sf;
}

ساكن مضمن HVX_ويڪٽر hvx_vec_reduce_جوڙ_f32x2(HVX_ويڪٽر in0, HVX_ويڪٽر in1) {
    HVX_ويڪٽر_Pair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_ويڪٽر  جوڙ_sf  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    جوڙ_sf = Q6_Vsf_vadd_VsfVsf(جوڙ_sf, Q6_V_vror_VR(جوڙ_sf, VLEN / 2));
    جوڙ_sf = Q6_Vsf_vadd_VsfVsf(جوڙ_sf, Q6_V_vror_VR(جوڙ_sf, VLEN / 4));
    جوڙ_sf = Q6_Vsf_vadd_VsfVsf(جوڙ_sf, Q6_V_vror_VR(جوڙ_sf, VLEN / 8));
    جوڙ_sf = Q6_Vsf_vadd_VsfVsf(جوڙ_sf, Q6_V_vror_VR(جوڙ_sf, VLEN / 16));
    أرجع جوڙ_sf;
}

ساكن مضمن HVX_ويڪٽر hvx_vec_reduce_جوڙ_n_f32(HVX_ويڪٽر اندريون, بنانشان صحيح n) {
    بنانشان صحيح ڪل = n * 4;  // total vec nbytes
    بنانشان صحيح ويڪر = 4;      // fp32 nbytes

    HVX_ويڪٽر جوڙ = اندريون, جوڙ_t;
    طالما (ويڪر < ڪل) {
        جوڙ_t = Q6_V_vror_VR(جوڙ, ويڪر);       // rotate right
        جوڙ   = Q6_Vsf_vadd_VsfVsf(جوڙ, جوڙ_t); // elementwise sum
        ويڪر = ويڪر << 1;
    }
    أرجع جوڙ;
}

#وإلا

ساكن مضمن HVX_ويڪٽر hvx_vec_reduce_جوڙ_f32x4(HVX_ويڪٽر_x4 اندريون) {
    HVX_ويڪٽر_Pair جوڙ_p01  = Q6_W_vshuff_VVR(اندريون.v[1], اندريون.v[0], 4);
    HVX_ويڪٽر_Pair جوڙ_p23  = Q6_W_vshuff_VVR(اندريون.v[3], اندريون.v[2], 4);
    HVX_ويڪٽر     جوڙ_qf01 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(جوڙ_p01), Q6_V_hi_W(جوڙ_p01));
    HVX_ويڪٽر     جوڙ_qf23 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(جوڙ_p23), Q6_V_hi_W(جوڙ_p23));

    HVX_ويڪٽر_Pair جوڙ_p0123 = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(جوڙ_qf23), Q6_Vsf_equals_Vqf32(جوڙ_qf01), 8);
    HVX_ويڪٽر     جوڙ_qf    = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(جوڙ_p0123), Q6_V_hi_W(جوڙ_p0123));

    جوڙ_qf = Q6_Vqf32_vadd_Vqf32Vsf(جوڙ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(جوڙ_qf), VLEN / 2));
    جوڙ_qf = Q6_Vqf32_vadd_Vqf32Vsf(جوڙ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(جوڙ_qf), VLEN / 4));
    جوڙ_qf = Q6_Vqf32_vadd_Vqf32Vsf(جوڙ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(جوڙ_qf), VLEN / 8));
    أرجع Q6_Vsf_equals_Vqf32(جوڙ_qf);
}

ساكن مضمن HVX_ويڪٽر hvx_vec_reduce_جوڙ_f32x2(HVX_ويڪٽر in0, HVX_ويڪٽر in1) {
    HVX_ويڪٽر_Pair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_ويڪٽر  جوڙ_qf  = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    جوڙ_qf = Q6_Vqf32_vadd_Vqf32Vsf(جوڙ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(جوڙ_qf), VLEN / 2));
    جوڙ_qf = Q6_Vqf32_vadd_Vqf32Vsf(جوڙ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(جوڙ_qf), VLEN / 4));
    جوڙ_qf = Q6_Vqf32_vadd_Vqf32Vsf(جوڙ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(جوڙ_qf), VLEN / 8));
    جوڙ_qf = Q6_Vqf32_vadd_Vqf32Vsf(جوڙ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(جوڙ_qf), VLEN / 16));
    أرجع Q6_Vsf_equals_Vqf32(جوڙ_qf);
}

ساكن مضمن HVX_ويڪٽر hvx_vec_reduce_جوڙ_n_f32(HVX_ويڪٽر اندريون, بنانشان صحيح n) {
    بنانشان صحيح ڪل = n * 4;  // total vec nbytes
    بنانشان صحيح ويڪر = 4;      // fp32 nbytes

    HVX_ويڪٽر جوڙ = اندريون, جوڙ_t;
    طالما (ويڪر < ڪل) {
        جوڙ_t = Q6_V_vror_VR(جوڙ, ويڪر);                               // rotate right
        جوڙ   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(جوڙ, جوڙ_t));  // elementwise sum
        ويڪر = ويڪر << 1;
    }
    أرجع جوڙ;
}

#جيڪڏهن_پورو

ساكن مضمن HVX_ويڪٽر hvx_vec_reduce_جوڙ_f32(HVX_ويڪٽر اندريون) {
    أرجع hvx_vec_reduce_جوڙ_n_f32(اندريون, 32);
}

ساكن مضمن HVX_ويڪٽر hvx_vec_reduce_max_f16(HVX_ويڪٽر اندريون) {
    بنانشان ڪل = 128;  // total vec nbytes
    بنانشان ويڪر = 2;    // fp16 nbytes

    HVX_ويڪٽر _max = اندريون, _max_t;
    طالما (ويڪر < ڪل) {
        _max_t = Q6_V_vror_VR(_max, ويڪر);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        ويڪر  = ويڪر << 1;
    }

    أرجع _max;
}

ساكن مضمن HVX_ويڪٽر hvx_vec_reduce_max2_f16(HVX_ويڪٽر اندريون, HVX_ويڪٽر _max) {
    بنانشان ڪل = 128;  // total vec nbytes
    بنانشان ويڪر = 2;    // fp32 nbytes

    HVX_ويڪٽر _max_t;

    _max = Q6_Vhf_vmax_VhfVhf(اندريون, _max);
    طالما (ويڪر < ڪل) {
        _max_t = Q6_V_vror_VR(_max, ويڪر);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        ويڪر  = ويڪر << 1;
    }

    أرجع _max;
}

ساكن مضمن HVX_ويڪٽر hvx_vec_reduce_max_f32(HVX_ويڪٽر اندريون) {
    بنانشان ڪل = 128;  // total vec nbytes
    بنانشان ويڪر = 4;    // fp32 nbytes

    HVX_ويڪٽر _max = اندريون, _max_t;
    طالما (ويڪر < ڪل) {
        _max_t = Q6_V_vror_VR(_max, ويڪر);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        ويڪر  = ويڪر << 1;
    }

    أرجع _max;
}

ساكن مضمن HVX_ويڪٽر hvx_vec_reduce_max2_f32(HVX_ويڪٽر اندريون, HVX_ويڪٽر _max) {
    بنانشان ڪل = 128;  // total vec nbytes
    بنانشان ويڪر = 4;    // fp32 nbytes

    HVX_ويڪٽر _max_t;

    _max = Q6_Vsf_vmax_VsfVsf(اندريون, _max);
    طالما (ويڪر < ڪل) {
        _max_t = Q6_V_vror_VR(_max, ويڪر);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        ويڪر  = ويڪر << 1;
    }

    أرجع _max;
}

#وصف_ڪريو hvx_reduce_loop_body(src_type, init_vec, pad_vec, vec_op, reduce_op, scalar_reduce) \
    افعل {                                                                                    \
        src_type * محدود vsrc = (src_type *) src;                                        \
        HVX_ويڪٽر acc = init_vec;                                                          \
                                                                                            \
        ثابت uint32_t elem_سائيز = حجم(عائم);                                           \
        ثابت uint32_t epv  = 128 / elem_سائيز;                                              \
        ثابت uint32_t nvec = انگ_elems / epv;                                              \
        ثابت uint32_t nloe = انگ_elems % epv;                                              \
                                                                                            \
        uint32_t i = 0;                                                                     \
        _Pragma("unroll(4)")                                                                \
        لكل (; i < nvec; i++) {                                                             \
            acc = vec_op(acc, vsrc[i]);                                                     \
        }                                                                                   \
        إذا (nloe) {                                                                         \
            ثابت عائم * srcf = (ثابت عائم *) src + i * epv;                             \
            HVX_ويڪٽر اندريون = *(HVX_UVector *) srcf;                                          \
            HVX_ويڪٽر عارضي = Q6_V_valign_VVR(اندريون, pad_vec, nloe * elem_سائيز);               \
            acc = vec_op(acc, عارضي);                                                        \
        }                                                                                   \
        HVX_ويڪٽر v = reduce_op(acc);                                                      \
        أرجع scalar_reduce(v);                                                            \
    } طالما(0)

#وصف_ڪريو HVX_REDUCE_MAX_OP(acc, val) Q6_Vsf_vmax_VsfVsf(acc, val)
#وصف_ڪريو HVX_REDUCE_جوڙ_OP(acc, val) Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(acc), val)
#وصف_ڪريو HVX_جوڙ_SQ_OP(acc, val) Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(val, val))
#وصف_ڪريو HVX_REDUCE_MAX_SCALAR(v) hvx_vec_get_f32(v)
#وصف_ڪريو HVX_REDUCE_جوڙ_SCALAR(v) hvx_vec_get_f32(Q6_Vsf_equals_Vqf32(v))

// Max variants

ساكن مضمن عائم hvx_reduce_max_f32_a(ثابت uint8_t * محدود src, ثابت صحيح انگ_elems) {
    HVX_ويڪٽر init_vec = hvx_vec_splat_f32(((ثابت عائم *) src)[0]);
    أكد((بنانشان ڊگهو) src % 128 == 0);
    hvx_reduce_loop_body(HVX_ويڪٽر, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

ساكن مضمن عائم hvx_reduce_max_f32_u(ثابت uint8_t * محدود src, ثابت صحيح انگ_elems) {
    HVX_ويڪٽر init_vec = hvx_vec_splat_f32(((ثابت عائم *) src)[0]);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

ساكن مضمن عائم hvx_reduce_max_f32(ثابت uint8_t * محدود src, ثابت صحيح انگ_elems) {
    إذا (hex_is_aligned((فراغ *) src, 128)) {
        أرجع hvx_reduce_max_f32_a(src, انگ_elems);
    } وإلا {
        أرجع hvx_reduce_max_f32_u(src, انگ_elems);
    }
}

// Sum variants

ساكن مضمن عائم hvx_reduce_جوڙ_f32_a(ثابت uint8_t * محدود src, ثابت صحيح انگ_elems) {
    HVX_ويڪٽر init_vec = Q6_V_vsplat_R(0);
    أكد((بنانشان ڊگهو) src % 128 == 0);
    hvx_reduce_loop_body(HVX_ويڪٽر, init_vec, init_vec, HVX_REDUCE_جوڙ_OP, hvx_vec_reduce_جوڙ_qf32, HVX_REDUCE_جوڙ_SCALAR);
}

ساكن مضمن عائم hvx_reduce_جوڙ_f32_u(ثابت uint8_t * محدود src, ثابت صحيح انگ_elems) {
    HVX_ويڪٽر init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_جوڙ_OP, hvx_vec_reduce_جوڙ_qf32, HVX_REDUCE_جوڙ_SCALAR);
}

ساكن مضمن عائم hvx_reduce_جوڙ_f32(ثابت uint8_t * محدود src, ثابت صحيح انگ_elems) {
    إذا (hex_is_aligned((فراغ *) src, 128)) {
        أرجع hvx_reduce_جوڙ_f32_a(src, انگ_elems);
    } وإلا {
        أرجع hvx_reduce_جوڙ_f32_u(src, انگ_elems);
    }
}

// Sum of squares variants

ساكن مضمن عائم hvx_جوڙ_جو_squares_f32_a(ثابت uint8_t * محدود src, ثابت صحيح انگ_elems) {
    HVX_ويڪٽر init_vec = Q6_V_vsplat_R(0);
    أكد((uintptr_t) src % 128 == 0);
    hvx_reduce_loop_body(HVX_ويڪٽر, init_vec, init_vec, HVX_جوڙ_SQ_OP, hvx_vec_reduce_جوڙ_qf32, HVX_REDUCE_جوڙ_SCALAR);
}

ساكن مضمن عائم hvx_جوڙ_جو_squares_f32_u(ثابت uint8_t * محدود src, ثابت صحيح انگ_elems) {
    HVX_ويڪٽر init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_جوڙ_SQ_OP, hvx_vec_reduce_جوڙ_qf32, HVX_REDUCE_جوڙ_SCALAR);
}

ساكن مضمن عائم hvx_جوڙ_جو_squares_f32(ثابت uint8_t * محدود src, ثابت صحيح انگ_elems) {
    إذا (hex_is_aligned((فراغ *) src, 128)) {
        أرجع hvx_جوڙ_جو_squares_f32_a(src, انگ_elems);
    } وإلا {
        أرجع hvx_جوڙ_جو_squares_f32_u(src, انگ_elems);
    }
}

// Signed 32-bit Integer Max variants

ساكن مضمن HVX_ويڪٽر hvx_vec_reduce_max_n_i32(HVX_ويڪٽر اندريون, بنانشان صحيح n) {
    بنانشان صحيح ڪل = n * 4;  // total vec nbytes
    بنانشان صحيح ويڪر = 4;      // int32 nbytes

    HVX_ويڪٽر max_val = اندريون, max_t;
    طالما (ويڪر < ڪل) {
        max_t = Q6_V_vror_VR(max_val, ويڪر);          // rotate right
        max_val = Q6_Vw_vmax_VwVw(max_t, max_val);      // elementwise signed max
        ويڪر = ويڪر << 1;
    }
    أرجع max_val;
}

ساكن مضمن HVX_ويڪٽر hvx_vec_reduce_max_i32(HVX_ويڪٽر اندريون) {
    أرجع hvx_vec_reduce_max_n_i32(اندريون, 32);
}

ساكن مضمن int32_t hvx_reduce_max_i32_a(ثابت uint8_t * محدود src, ثابت صحيح انگ_elems) {
    HVX_ويڪٽر init_vec = Q6_V_vsplat_R(((ثابت int32_t *) src)[0]);
    HVX_ويڪٽر pad_vec  = Q6_V_vsplat_R(0x80000000);
    أكد((uintptr_t) src % 128 == 0);
    hvx_reduce_loop_body(HVX_ويڪٽر, init_vec, pad_vec, Q6_Vw_vmax_VwVw, hvx_vec_reduce_max_i32, hvx_vec_get_i32);
}

ساكن مضمن int32_t hvx_reduce_max_i32_u(ثابت uint8_t * محدود src, ثابت صحيح انگ_elems) {
    HVX_ويڪٽر init_vec = Q6_V_vsplat_R(((ثابت int32_t *) src)[0]);
    HVX_ويڪٽر pad_vec  = Q6_V_vsplat_R(0x80000000);
    hvx_reduce_loop_body(HVX_UVector, init_vec, pad_vec, Q6_Vw_vmax_VwVw, hvx_vec_reduce_max_i32, hvx_vec_get_i32);
}

ساكن مضمن int32_t hvx_reduce_max_i32(ثابت uint8_t * محدود src, ثابت صحيح انگ_elems) {
    إذا (hex_is_aligned((فراغ *) src, 128)) {
        أرجع hvx_reduce_max_i32_a(src, انگ_elems);
    } وإلا {
        أرجع hvx_reduce_max_i32_u(src, انگ_elems);
    }
}

ساكن مضمن فراغ hvx_argmax_f32(
    ثابت عائم * محدود src,
    uint32_t n,
    uint32_t وٿي,
    عائم * ٻاهريون_val,
    int32_t * ٻاهريون_idx
) {
    إذا (n == 0) {
        *ٻاهريون_val = -INFINITY;
        *ٻاهريون_idx = (int32_t) وٿي;
        أرجع;
    }

    إذا (n < 32 || !hex_is_aligned((فراغ *) src, 128)) {
        عائم بهترين_val = src[0];
        int32_t بهترين_idx = (int32_t) وٿي;
        لكل (uint32_t i = 1; i < n; i++) {
            إذا (src[i] > بهترين_val) {
                بهترين_val = src[i];
                بهترين_idx = (int32_t) (وٿي + i);
            }
        }
        *ٻاهريون_val = بهترين_val;
        *ٻاهريون_idx = بهترين_idx;
        أرجع;
    }

    ساكن ثابت int32_t c_lane_idx[32] __attribute__((aligned(128))) = {
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    };

    ثابت HVX_ويڪٽر v_init_idx = *(ثابت HVX_ويڪٽر *) c_lane_idx;
    ثابت HVX_ويڪٽر v_قدم     = Q6_V_vsplat_R(32);
    HVX_ويڪٽر v_cur_idx        = Q6_Vw_vadd_VwVw(v_init_idx, Q6_V_vsplat_R((int32_t) وٿي));
    HVX_ويڪٽر v_max_val        = hvx_vec_splat_f32(-INFINITY);
    HVX_ويڪٽر v_max_idx        = v_cur_idx;

    ثابت HVX_ويڪٽر * vsrc = (ثابت HVX_ويڪٽر *) src;
    ثابت uint32_t nvec = n / 32;

    لكل (uint32_t vi = 0; vi < nvec; vi++) {
        HVX_ويڪٽر v = vsrc[vi];
        HVX_ويڪٽر_Pred pred = Q6_Q_vcmp_gt_VsfVsf(v, v_max_val);
        v_max_val = Q6_V_vmux_QVV(pred, v, v_max_val);
        v_max_idx = Q6_V_vmux_QVV(pred, v_cur_idx, v_max_idx);
        v_cur_idx = Q6_Vw_vadd_VwVw(v_cur_idx, v_قدم);
    }

    HVX_ويڪٽر_Alias u_val, u_idx;
    u_val.v = v_max_val;
    u_idx.v = v_max_idx;

    عائم بهترين_val = u_val.fp32[0];
    int32_t بهترين_idx = (int32_t) u_idx.w[0];
    لكل (صحيح i = 1; i < 32; i++) {
        إذا (u_val.fp32[i] > بهترين_val) {
            بهترين_val = u_val.fp32[i];
            بهترين_idx = (int32_t) u_idx.w[i];
        }
    }

    لكل (uint32_t i = nvec * 32; i < n; i++) {
        إذا (src[i] > بهترين_val) {
            بهترين_val = src[i];
            بهترين_idx = (int32_t) (وٿي + i);
        }
    }

    *ٻاهريون_val = بهترين_val;
    *ٻاهريون_idx = بهترين_idx;
}

#وصف_رد hvx_reduce_loop_body
#وصف_رد HVX_REDUCE_MAX_OP
#وصف_رد HVX_REDUCE_جوڙ_OP
#وصف_رد HVX_REDUCE_MAX_SCALAR
#وصف_رد HVX_REDUCE_جوڙ_SCALAR
#وصف_رد HVX_جوڙ_SQ_OP

#جيڪڏهن_پورو /* HVX_REDUCE_H */
