#که_نه_تعریف_شوی HVX_REDUCE_H
#تعریف_کړه HVX_REDUCE_H

#ضمن <math.h>
#ضمن <stdbool.h>
#ضمن <stdint.h>
#ضمن <assert.h>

#ضمن "hex-utils.h"
#ضمن "hvx-base.h"
#ضمن "hvx-types.h"

ساكن مضمن HVX_ویکټور hvx_vec_reduce_مجموعه_n_i32(HVX_ویکټور دننه, بینښه صحيح n) {
    بینښه صحيح ټول = n * 4;  // total vec nbytes
    بینښه صحيح پلنوالی = 4;      // int32

    HVX_ویکټور مجموعه = دننه, مجموعه_t;
    طالما (پلنوالی < ټول) {
        مجموعه_t = Q6_V_vror_VR(مجموعه, پلنوالی);     // rotate right
        مجموعه   = Q6_Vw_vadd_VwVw(مجموعه_t, مجموعه);  // elementwise sum
        پلنوالی = پلنوالی << 1;
    }
    أرجع مجموعه;
}

ساكن مضمن HVX_ویکټور hvx_vec_reduce_مجموعه_i32(HVX_ویکټور دننه) {
    أرجع hvx_vec_reduce_مجموعه_n_i32(دننه, 32);
}

ساكن مضمن HVX_ویکټور hvx_vec_reduce_مجموعه_n_qf32(HVX_ویکټور دننه, بینښه صحيح n) {
    بینښه صحيح ټول = n * 4;  // total vec nbytes
    بینښه صحيح پلنوالی = 4;      // fp32 nbytes

    HVX_ویکټور مجموعه = دننه, مجموعه_t;
    طالما (پلنوالی < ټول) {
        مجموعه_t = Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(مجموعه), پلنوالی);  // rotate right
        مجموعه   = Q6_Vqf32_vadd_Vqf32Vsf(مجموعه, مجموعه_t);             // elementwise sum
        پلنوالی = پلنوالی << 1;
    }
    أرجع مجموعه;
}

ساكن مضمن HVX_ویکټور hvx_vec_reduce_مجموعه_qf32(HVX_ویکټور دننه) {
    أرجع hvx_vec_reduce_مجموعه_n_qf32(دننه, 32);
}

#إذا __HVX_ARCH__ > 75

ساكن مضمن HVX_ویکټور hvx_vec_reduce_مجموعه_f32x4(HVX_ویکټور_x4 دننه) {
    HVX_ویکټور_Pair مجموعه_p01 = Q6_W_vshuff_VVR(دننه.v[1], دننه.v[0], 4);
    HVX_ویکټور_Pair مجموعه_p23 = Q6_W_vshuff_VVR(دننه.v[3], دننه.v[2], 4);
    HVX_ویکټور  مجموعه_sf01  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(مجموعه_p01), Q6_V_hi_W(مجموعه_p01));
    HVX_ویکټور  مجموعه_sf23  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(مجموعه_p23), Q6_V_hi_W(مجموعه_p23));

    HVX_ویکټور_Pair مجموعه_p0123 = Q6_W_vshuff_VVR(مجموعه_sf23, مجموعه_sf01, 8);
    HVX_ویکټور  مجموعه_sf       = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(مجموعه_p0123), Q6_V_hi_W(مجموعه_p0123));

    مجموعه_sf = Q6_Vsf_vadd_VsfVsf(مجموعه_sf, Q6_V_vror_VR(مجموعه_sf, VLEN / 2));
    مجموعه_sf = Q6_Vsf_vadd_VsfVsf(مجموعه_sf, Q6_V_vror_VR(مجموعه_sf, VLEN / 4));
    مجموعه_sf = Q6_Vsf_vadd_VsfVsf(مجموعه_sf, Q6_V_vror_VR(مجموعه_sf, VLEN / 8));
    أرجع مجموعه_sf;
}

ساكن مضمن HVX_ویکټور hvx_vec_reduce_مجموعه_f32x2(HVX_ویکټور in0, HVX_ویکټور in1) {
    HVX_ویکټور_Pair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_ویکټور  مجموعه_sf  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    مجموعه_sf = Q6_Vsf_vadd_VsfVsf(مجموعه_sf, Q6_V_vror_VR(مجموعه_sf, VLEN / 2));
    مجموعه_sf = Q6_Vsf_vadd_VsfVsf(مجموعه_sf, Q6_V_vror_VR(مجموعه_sf, VLEN / 4));
    مجموعه_sf = Q6_Vsf_vadd_VsfVsf(مجموعه_sf, Q6_V_vror_VR(مجموعه_sf, VLEN / 8));
    مجموعه_sf = Q6_Vsf_vadd_VsfVsf(مجموعه_sf, Q6_V_vror_VR(مجموعه_sf, VLEN / 16));
    أرجع مجموعه_sf;
}

ساكن مضمن HVX_ویکټور hvx_vec_reduce_مجموعه_n_f32(HVX_ویکټور دننه, بینښه صحيح n) {
    بینښه صحيح ټول = n * 4;  // total vec nbytes
    بینښه صحيح پلنوالی = 4;      // fp32 nbytes

    HVX_ویکټور مجموعه = دننه, مجموعه_t;
    طالما (پلنوالی < ټول) {
        مجموعه_t = Q6_V_vror_VR(مجموعه, پلنوالی);       // rotate right
        مجموعه   = Q6_Vsf_vadd_VsfVsf(مجموعه, مجموعه_t); // elementwise sum
        پلنوالی = پلنوالی << 1;
    }
    أرجع مجموعه;
}

#وإلا

ساكن مضمن HVX_ویکټور hvx_vec_reduce_مجموعه_f32x4(HVX_ویکټور_x4 دننه) {
    HVX_ویکټور_Pair مجموعه_p01  = Q6_W_vshuff_VVR(دننه.v[1], دننه.v[0], 4);
    HVX_ویکټور_Pair مجموعه_p23  = Q6_W_vshuff_VVR(دننه.v[3], دننه.v[2], 4);
    HVX_ویکټور     مجموعه_qf01 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(مجموعه_p01), Q6_V_hi_W(مجموعه_p01));
    HVX_ویکټور     مجموعه_qf23 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(مجموعه_p23), Q6_V_hi_W(مجموعه_p23));

    HVX_ویکټور_Pair مجموعه_p0123 = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(مجموعه_qf23), Q6_Vsf_equals_Vqf32(مجموعه_qf01), 8);
    HVX_ویکټور     مجموعه_qf    = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(مجموعه_p0123), Q6_V_hi_W(مجموعه_p0123));

    مجموعه_qf = Q6_Vqf32_vadd_Vqf32Vsf(مجموعه_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(مجموعه_qf), VLEN / 2));
    مجموعه_qf = Q6_Vqf32_vadd_Vqf32Vsf(مجموعه_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(مجموعه_qf), VLEN / 4));
    مجموعه_qf = Q6_Vqf32_vadd_Vqf32Vsf(مجموعه_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(مجموعه_qf), VLEN / 8));
    أرجع Q6_Vsf_equals_Vqf32(مجموعه_qf);
}

ساكن مضمن HVX_ویکټور hvx_vec_reduce_مجموعه_f32x2(HVX_ویکټور in0, HVX_ویکټور in1) {
    HVX_ویکټور_Pair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_ویکټور  مجموعه_qf  = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    مجموعه_qf = Q6_Vqf32_vadd_Vqf32Vsf(مجموعه_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(مجموعه_qf), VLEN / 2));
    مجموعه_qf = Q6_Vqf32_vadd_Vqf32Vsf(مجموعه_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(مجموعه_qf), VLEN / 4));
    مجموعه_qf = Q6_Vqf32_vadd_Vqf32Vsf(مجموعه_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(مجموعه_qf), VLEN / 8));
    مجموعه_qf = Q6_Vqf32_vadd_Vqf32Vsf(مجموعه_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(مجموعه_qf), VLEN / 16));
    أرجع Q6_Vsf_equals_Vqf32(مجموعه_qf);
}

ساكن مضمن HVX_ویکټور hvx_vec_reduce_مجموعه_n_f32(HVX_ویکټور دننه, بینښه صحيح n) {
    بینښه صحيح ټول = n * 4;  // total vec nbytes
    بینښه صحيح پلنوالی = 4;      // fp32 nbytes

    HVX_ویکټور مجموعه = دننه, مجموعه_t;
    طالما (پلنوالی < ټول) {
        مجموعه_t = Q6_V_vror_VR(مجموعه, پلنوالی);                               // rotate right
        مجموعه   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(مجموعه, مجموعه_t));  // elementwise sum
        پلنوالی = پلنوالی << 1;
    }
    أرجع مجموعه;
}

#د_که_پای

ساكن مضمن HVX_ویکټور hvx_vec_reduce_مجموعه_f32(HVX_ویکټور دننه) {
    أرجع hvx_vec_reduce_مجموعه_n_f32(دننه, 32);
}

ساكن مضمن HVX_ویکټور hvx_vec_reduce_max_f16(HVX_ویکټور دننه) {
    بینښه ټول = 128;  // total vec nbytes
    بینښه پلنوالی = 2;    // fp16 nbytes

    HVX_ویکټور _max = دننه, _max_t;
    طالما (پلنوالی < ټول) {
        _max_t = Q6_V_vror_VR(_max, پلنوالی);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        پلنوالی  = پلنوالی << 1;
    }

    أرجع _max;
}

ساكن مضمن HVX_ویکټور hvx_vec_reduce_max2_f16(HVX_ویکټور دننه, HVX_ویکټور _max) {
    بینښه ټول = 128;  // total vec nbytes
    بینښه پلنوالی = 2;    // fp32 nbytes

    HVX_ویکټور _max_t;

    _max = Q6_Vhf_vmax_VhfVhf(دننه, _max);
    طالما (پلنوالی < ټول) {
        _max_t = Q6_V_vror_VR(_max, پلنوالی);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        پلنوالی  = پلنوالی << 1;
    }

    أرجع _max;
}

ساكن مضمن HVX_ویکټور hvx_vec_reduce_max_f32(HVX_ویکټور دننه) {
    بینښه ټول = 128;  // total vec nbytes
    بینښه پلنوالی = 4;    // fp32 nbytes

    HVX_ویکټور _max = دننه, _max_t;
    طالما (پلنوالی < ټول) {
        _max_t = Q6_V_vror_VR(_max, پلنوالی);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        پلنوالی  = پلنوالی << 1;
    }

    أرجع _max;
}

ساكن مضمن HVX_ویکټور hvx_vec_reduce_max2_f32(HVX_ویکټور دننه, HVX_ویکټور _max) {
    بینښه ټول = 128;  // total vec nbytes
    بینښه پلنوالی = 4;    // fp32 nbytes

    HVX_ویکټور _max_t;

    _max = Q6_Vsf_vmax_VsfVsf(دننه, _max);
    طالما (پلنوالی < ټول) {
        _max_t = Q6_V_vror_VR(_max, پلنوالی);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        پلنوالی  = پلنوالی << 1;
    }

    أرجع _max;
}

#تعریف_کړه hvx_reduce_loop_body(src_type, init_vec, pad_vec, vec_op, reduce_op, scalar_reduce) \
    افعل {                                                                                    \
        src_type * محدود vsrc = (src_type *) src;                                        \
        HVX_ویکټور acc = init_vec;                                                          \
                                                                                            \
        ثابت uint32_t elem_اندازه = حجم(عائم);                                           \
        ثابت uint32_t epv  = 128 / elem_اندازه;                                              \
        ثابت uint32_t nvec = عدد_elems / epv;                                              \
        ثابت uint32_t nloe = عدد_elems % epv;                                              \
                                                                                            \
        uint32_t i = 0;                                                                     \
        _Pragma("unroll(4)")                                                                \
        لكل (; i < nvec; i++) {                                                             \
            acc = vec_op(acc, vsrc[i]);                                                     \
        }                                                                                   \
        إذا (nloe) {                                                                         \
            ثابت عائم * srcf = (ثابت عائم *) src + i * epv;                             \
            HVX_ویکټور دننه = *(HVX_UVector *) srcf;                                          \
            HVX_ویکټور لنډمهاله = Q6_V_valign_VVR(دننه, pad_vec, nloe * elem_اندازه);               \
            acc = vec_op(acc, لنډمهاله);                                                        \
        }                                                                                   \
        HVX_ویکټور v = reduce_op(acc);                                                      \
        أرجع scalar_reduce(v);                                                            \
    } طالما(0)

#تعریف_کړه HVX_REDUCE_MAX_OP(acc, val) Q6_Vsf_vmax_VsfVsf(acc, val)
#تعریف_کړه HVX_REDUCE_مجموعه_OP(acc, val) Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(acc), val)
#تعریف_کړه HVX_مجموعه_SQ_OP(acc, val) Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(val, val))
#تعریف_کړه HVX_REDUCE_MAX_SCALAR(v) hvx_vec_get_f32(v)
#تعریف_کړه HVX_REDUCE_مجموعه_SCALAR(v) hvx_vec_get_f32(Q6_Vsf_equals_Vqf32(v))

// Max variants

ساكن مضمن عائم hvx_reduce_max_f32_a(ثابت uint8_t * محدود src, ثابت صحيح عدد_elems) {
    HVX_ویکټور init_vec = hvx_vec_splat_f32(((ثابت عائم *) src)[0]);
    أكد((بینښه اوږد) src % 128 == 0);
    hvx_reduce_loop_body(HVX_ویکټور, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

ساكن مضمن عائم hvx_reduce_max_f32_u(ثابت uint8_t * محدود src, ثابت صحيح عدد_elems) {
    HVX_ویکټور init_vec = hvx_vec_splat_f32(((ثابت عائم *) src)[0]);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

ساكن مضمن عائم hvx_reduce_max_f32(ثابت uint8_t * محدود src, ثابت صحيح عدد_elems) {
    إذا (hex_is_aligned((فراغ *) src, 128)) {
        أرجع hvx_reduce_max_f32_a(src, عدد_elems);
    } وإلا {
        أرجع hvx_reduce_max_f32_u(src, عدد_elems);
    }
}

// Sum variants

ساكن مضمن عائم hvx_reduce_مجموعه_f32_a(ثابت uint8_t * محدود src, ثابت صحيح عدد_elems) {
    HVX_ویکټور init_vec = Q6_V_vsplat_R(0);
    أكد((بینښه اوږد) src % 128 == 0);
    hvx_reduce_loop_body(HVX_ویکټور, init_vec, init_vec, HVX_REDUCE_مجموعه_OP, hvx_vec_reduce_مجموعه_qf32, HVX_REDUCE_مجموعه_SCALAR);
}

ساكن مضمن عائم hvx_reduce_مجموعه_f32_u(ثابت uint8_t * محدود src, ثابت صحيح عدد_elems) {
    HVX_ویکټور init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_مجموعه_OP, hvx_vec_reduce_مجموعه_qf32, HVX_REDUCE_مجموعه_SCALAR);
}

ساكن مضمن عائم hvx_reduce_مجموعه_f32(ثابت uint8_t * محدود src, ثابت صحيح عدد_elems) {
    إذا (hex_is_aligned((فراغ *) src, 128)) {
        أرجع hvx_reduce_مجموعه_f32_a(src, عدد_elems);
    } وإلا {
        أرجع hvx_reduce_مجموعه_f32_u(src, عدد_elems);
    }
}

// Sum of squares variants

ساكن مضمن عائم hvx_مجموعه_د_squares_f32_a(ثابت uint8_t * محدود src, ثابت صحيح عدد_elems) {
    HVX_ویکټور init_vec = Q6_V_vsplat_R(0);
    أكد((uintptr_t) src % 128 == 0);
    hvx_reduce_loop_body(HVX_ویکټور, init_vec, init_vec, HVX_مجموعه_SQ_OP, hvx_vec_reduce_مجموعه_qf32, HVX_REDUCE_مجموعه_SCALAR);
}

ساكن مضمن عائم hvx_مجموعه_د_squares_f32_u(ثابت uint8_t * محدود src, ثابت صحيح عدد_elems) {
    HVX_ویکټور init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_مجموعه_SQ_OP, hvx_vec_reduce_مجموعه_qf32, HVX_REDUCE_مجموعه_SCALAR);
}

ساكن مضمن عائم hvx_مجموعه_د_squares_f32(ثابت uint8_t * محدود src, ثابت صحيح عدد_elems) {
    إذا (hex_is_aligned((فراغ *) src, 128)) {
        أرجع hvx_مجموعه_د_squares_f32_a(src, عدد_elems);
    } وإلا {
        أرجع hvx_مجموعه_د_squares_f32_u(src, عدد_elems);
    }
}

// Signed 32-bit Integer Max variants

ساكن مضمن HVX_ویکټور hvx_vec_reduce_max_n_i32(HVX_ویکټور دننه, بینښه صحيح n) {
    بینښه صحيح ټول = n * 4;  // total vec nbytes
    بینښه صحيح پلنوالی = 4;      // int32 nbytes

    HVX_ویکټور max_val = دننه, max_t;
    طالما (پلنوالی < ټول) {
        max_t = Q6_V_vror_VR(max_val, پلنوالی);          // rotate right
        max_val = Q6_Vw_vmax_VwVw(max_t, max_val);      // elementwise signed max
        پلنوالی = پلنوالی << 1;
    }
    أرجع max_val;
}

ساكن مضمن HVX_ویکټور hvx_vec_reduce_max_i32(HVX_ویکټور دننه) {
    أرجع hvx_vec_reduce_max_n_i32(دننه, 32);
}

ساكن مضمن int32_t hvx_reduce_max_i32_a(ثابت uint8_t * محدود src, ثابت صحيح عدد_elems) {
    HVX_ویکټور init_vec = Q6_V_vsplat_R(((ثابت int32_t *) src)[0]);
    HVX_ویکټور pad_vec  = Q6_V_vsplat_R(0x80000000);
    أكد((uintptr_t) src % 128 == 0);
    hvx_reduce_loop_body(HVX_ویکټور, init_vec, pad_vec, Q6_Vw_vmax_VwVw, hvx_vec_reduce_max_i32, hvx_vec_get_i32);
}

ساكن مضمن int32_t hvx_reduce_max_i32_u(ثابت uint8_t * محدود src, ثابت صحيح عدد_elems) {
    HVX_ویکټور init_vec = Q6_V_vsplat_R(((ثابت int32_t *) src)[0]);
    HVX_ویکټور pad_vec  = Q6_V_vsplat_R(0x80000000);
    hvx_reduce_loop_body(HVX_UVector, init_vec, pad_vec, Q6_Vw_vmax_VwVw, hvx_vec_reduce_max_i32, hvx_vec_get_i32);
}

ساكن مضمن int32_t hvx_reduce_max_i32(ثابت uint8_t * محدود src, ثابت صحيح عدد_elems) {
    إذا (hex_is_aligned((فراغ *) src, 128)) {
        أرجع hvx_reduce_max_i32_a(src, عدد_elems);
    } وإلا {
        أرجع hvx_reduce_max_i32_u(src, عدد_elems);
    }
}

ساكن مضمن فراغ hvx_argmax_f32(
    ثابت عائم * محدود src,
    uint32_t n,
    uint32_t دننه_واټن,
    عائم * بهر_val,
    int32_t * بهر_idx
) {
    إذا (n == 0) {
        *بهر_val = -INFINITY;
        *بهر_idx = (int32_t) دننه_واټن;
        أرجع;
    }

    إذا (n < 32 || !hex_is_aligned((فراغ *) src, 128)) {
        عائم غوره_val = src[0];
        int32_t غوره_idx = (int32_t) دننه_واټن;
        لكل (uint32_t i = 1; i < n; i++) {
            إذا (src[i] > غوره_val) {
                غوره_val = src[i];
                غوره_idx = (int32_t) (دننه_واټن + i);
            }
        }
        *بهر_val = غوره_val;
        *بهر_idx = غوره_idx;
        أرجع;
    }

    ساكن ثابت int32_t c_lane_idx[32] __attribute__((aligned(128))) = {
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    };

    ثابت HVX_ویکټور v_init_idx = *(ثابت HVX_ویکټور *) c_lane_idx;
    ثابت HVX_ویکټور v_ګام     = Q6_V_vsplat_R(32);
    HVX_ویکټور v_cur_idx        = Q6_Vw_vadd_VwVw(v_init_idx, Q6_V_vsplat_R((int32_t) دننه_واټن));
    HVX_ویکټور v_max_val        = hvx_vec_splat_f32(-INFINITY);
    HVX_ویکټور v_max_idx        = v_cur_idx;

    ثابت HVX_ویکټور * vsrc = (ثابت HVX_ویکټور *) src;
    ثابت uint32_t nvec = n / 32;

    لكل (uint32_t vi = 0; vi < nvec; vi++) {
        HVX_ویکټور v = vsrc[vi];
        HVX_ویکټور_Pred pred = Q6_Q_vcmp_gt_VsfVsf(v, v_max_val);
        v_max_val = Q6_V_vmux_QVV(pred, v, v_max_val);
        v_max_idx = Q6_V_vmux_QVV(pred, v_cur_idx, v_max_idx);
        v_cur_idx = Q6_Vw_vadd_VwVw(v_cur_idx, v_ګام);
    }

    HVX_ویکټور_Alias u_val, u_idx;
    u_val.v = v_max_val;
    u_idx.v = v_max_idx;

    عائم غوره_val = u_val.fp32[0];
    int32_t غوره_idx = (int32_t) u_idx.w[0];
    لكل (صحيح i = 1; i < 32; i++) {
        إذا (u_val.fp32[i] > غوره_val) {
            غوره_val = u_val.fp32[i];
            غوره_idx = (int32_t) u_idx.w[i];
        }
    }

    لكل (uint32_t i = nvec * 32; i < n; i++) {
        إذا (src[i] > غوره_val) {
            غوره_val = src[i];
            غوره_idx = (int32_t) (دننه_واټن + i);
        }
    }

    *بهر_val = غوره_val;
    *بهر_idx = غوره_idx;
}

#تعریف_لغوه hvx_reduce_loop_body
#تعریف_لغوه HVX_REDUCE_MAX_OP
#تعریف_لغوه HVX_REDUCE_مجموعه_OP
#تعریف_لغوه HVX_REDUCE_MAX_SCALAR
#تعریف_لغوه HVX_REDUCE_مجموعه_SCALAR
#تعریف_لغوه HVX_مجموعه_SQ_OP

#د_که_پای /* HVX_REDUCE_H */
