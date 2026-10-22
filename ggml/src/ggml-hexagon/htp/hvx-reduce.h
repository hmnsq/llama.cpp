#اگر_تعریف_نشده HVX_REDUCE_H
#تعریف_کن HVX_REDUCE_H

#ضمن <math.h>
#ضمن <stdbool.h>
#ضمن <stdint.h>
#ضمن <assert.h>

#ضمن "hex-utils.h"
#ضمن "hvx-base.h"
#ضمن "hvx-types.h"

ساكن مضمن HVX_وکتور hvx_vec_reduce_جمع_n_i32(HVX_وکتور درون, بیعلامت صحيح n) {
    بیعلامت صحيح مجموع = n * 4;  // total vec nbytes
    بیعلامت صحيح عرض = 4;      // int32

    HVX_وکتور جمع = درون, جمع_t;
    طالما (عرض < مجموع) {
        جمع_t = Q6_V_vror_VR(جمع, عرض);     // rotate right
        جمع   = Q6_Vw_vadd_VwVw(جمع_t, جمع);  // elementwise sum
        عرض = عرض << 1;
    }
    أرجع جمع;
}

ساكن مضمن HVX_وکتور hvx_vec_reduce_جمع_i32(HVX_وکتور درون) {
    أرجع hvx_vec_reduce_جمع_n_i32(درون, 32);
}

ساكن مضمن HVX_وکتور hvx_vec_reduce_جمع_n_qf32(HVX_وکتور درون, بیعلامت صحيح n) {
    بیعلامت صحيح مجموع = n * 4;  // total vec nbytes
    بیعلامت صحيح عرض = 4;      // fp32 nbytes

    HVX_وکتور جمع = درون, جمع_t;
    طالما (عرض < مجموع) {
        جمع_t = Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(جمع), عرض);  // rotate right
        جمع   = Q6_Vqf32_vadd_Vqf32Vsf(جمع, جمع_t);             // elementwise sum
        عرض = عرض << 1;
    }
    أرجع جمع;
}

ساكن مضمن HVX_وکتور hvx_vec_reduce_جمع_qf32(HVX_وکتور درون) {
    أرجع hvx_vec_reduce_جمع_n_qf32(درون, 32);
}

#إذا __HVX_ARCH__ > 75

ساكن مضمن HVX_وکتور hvx_vec_reduce_جمع_f32x4(HVX_وکتور_x4 درون) {
    HVX_وکتور_Pair جمع_p01 = Q6_W_vshuff_VVR(درون.v[1], درون.v[0], 4);
    HVX_وکتور_Pair جمع_p23 = Q6_W_vshuff_VVR(درون.v[3], درون.v[2], 4);
    HVX_وکتور  جمع_sf01  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(جمع_p01), Q6_V_hi_W(جمع_p01));
    HVX_وکتور  جمع_sf23  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(جمع_p23), Q6_V_hi_W(جمع_p23));

    HVX_وکتور_Pair جمع_p0123 = Q6_W_vshuff_VVR(جمع_sf23, جمع_sf01, 8);
    HVX_وکتور  جمع_sf       = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(جمع_p0123), Q6_V_hi_W(جمع_p0123));

    جمع_sf = Q6_Vsf_vadd_VsfVsf(جمع_sf, Q6_V_vror_VR(جمع_sf, VLEN / 2));
    جمع_sf = Q6_Vsf_vadd_VsfVsf(جمع_sf, Q6_V_vror_VR(جمع_sf, VLEN / 4));
    جمع_sf = Q6_Vsf_vadd_VsfVsf(جمع_sf, Q6_V_vror_VR(جمع_sf, VLEN / 8));
    أرجع جمع_sf;
}

ساكن مضمن HVX_وکتور hvx_vec_reduce_جمع_f32x2(HVX_وکتور in0, HVX_وکتور in1) {
    HVX_وکتور_Pair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_وکتور  جمع_sf  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    جمع_sf = Q6_Vsf_vadd_VsfVsf(جمع_sf, Q6_V_vror_VR(جمع_sf, VLEN / 2));
    جمع_sf = Q6_Vsf_vadd_VsfVsf(جمع_sf, Q6_V_vror_VR(جمع_sf, VLEN / 4));
    جمع_sf = Q6_Vsf_vadd_VsfVsf(جمع_sf, Q6_V_vror_VR(جمع_sf, VLEN / 8));
    جمع_sf = Q6_Vsf_vadd_VsfVsf(جمع_sf, Q6_V_vror_VR(جمع_sf, VLEN / 16));
    أرجع جمع_sf;
}

ساكن مضمن HVX_وکتور hvx_vec_reduce_جمع_n_f32(HVX_وکتور درون, بیعلامت صحيح n) {
    بیعلامت صحيح مجموع = n * 4;  // total vec nbytes
    بیعلامت صحيح عرض = 4;      // fp32 nbytes

    HVX_وکتور جمع = درون, جمع_t;
    طالما (عرض < مجموع) {
        جمع_t = Q6_V_vror_VR(جمع, عرض);       // rotate right
        جمع   = Q6_Vsf_vadd_VsfVsf(جمع, جمع_t); // elementwise sum
        عرض = عرض << 1;
    }
    أرجع جمع;
}

#وإلا

ساكن مضمن HVX_وکتور hvx_vec_reduce_جمع_f32x4(HVX_وکتور_x4 درون) {
    HVX_وکتور_Pair جمع_p01  = Q6_W_vshuff_VVR(درون.v[1], درون.v[0], 4);
    HVX_وکتور_Pair جمع_p23  = Q6_W_vshuff_VVR(درون.v[3], درون.v[2], 4);
    HVX_وکتور     جمع_qf01 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(جمع_p01), Q6_V_hi_W(جمع_p01));
    HVX_وکتور     جمع_qf23 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(جمع_p23), Q6_V_hi_W(جمع_p23));

    HVX_وکتور_Pair جمع_p0123 = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(جمع_qf23), Q6_Vsf_equals_Vqf32(جمع_qf01), 8);
    HVX_وکتور     جمع_qf    = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(جمع_p0123), Q6_V_hi_W(جمع_p0123));

    جمع_qf = Q6_Vqf32_vadd_Vqf32Vsf(جمع_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(جمع_qf), VLEN / 2));
    جمع_qf = Q6_Vqf32_vadd_Vqf32Vsf(جمع_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(جمع_qf), VLEN / 4));
    جمع_qf = Q6_Vqf32_vadd_Vqf32Vsf(جمع_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(جمع_qf), VLEN / 8));
    أرجع Q6_Vsf_equals_Vqf32(جمع_qf);
}

ساكن مضمن HVX_وکتور hvx_vec_reduce_جمع_f32x2(HVX_وکتور in0, HVX_وکتور in1) {
    HVX_وکتور_Pair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_وکتور  جمع_qf  = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    جمع_qf = Q6_Vqf32_vadd_Vqf32Vsf(جمع_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(جمع_qf), VLEN / 2));
    جمع_qf = Q6_Vqf32_vadd_Vqf32Vsf(جمع_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(جمع_qf), VLEN / 4));
    جمع_qf = Q6_Vqf32_vadd_Vqf32Vsf(جمع_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(جمع_qf), VLEN / 8));
    جمع_qf = Q6_Vqf32_vadd_Vqf32Vsf(جمع_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(جمع_qf), VLEN / 16));
    أرجع Q6_Vsf_equals_Vqf32(جمع_qf);
}

ساكن مضمن HVX_وکتور hvx_vec_reduce_جمع_n_f32(HVX_وکتور درون, بیعلامت صحيح n) {
    بیعلامت صحيح مجموع = n * 4;  // total vec nbytes
    بیعلامت صحيح عرض = 4;      // fp32 nbytes

    HVX_وکتور جمع = درون, جمع_t;
    طالما (عرض < مجموع) {
        جمع_t = Q6_V_vror_VR(جمع, عرض);                               // rotate right
        جمع   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(جمع, جمع_t));  // elementwise sum
        عرض = عرض << 1;
    }
    أرجع جمع;
}

#پایان_اگر

ساكن مضمن HVX_وکتور hvx_vec_reduce_جمع_f32(HVX_وکتور درون) {
    أرجع hvx_vec_reduce_جمع_n_f32(درون, 32);
}

ساكن مضمن HVX_وکتور hvx_vec_reduce_max_f16(HVX_وکتور درون) {
    بیعلامت مجموع = 128;  // total vec nbytes
    بیعلامت عرض = 2;    // fp16 nbytes

    HVX_وکتور _max = درون, _max_t;
    طالما (عرض < مجموع) {
        _max_t = Q6_V_vror_VR(_max, عرض);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        عرض  = عرض << 1;
    }

    أرجع _max;
}

ساكن مضمن HVX_وکتور hvx_vec_reduce_max2_f16(HVX_وکتور درون, HVX_وکتور _max) {
    بیعلامت مجموع = 128;  // total vec nbytes
    بیعلامت عرض = 2;    // fp32 nbytes

    HVX_وکتور _max_t;

    _max = Q6_Vhf_vmax_VhfVhf(درون, _max);
    طالما (عرض < مجموع) {
        _max_t = Q6_V_vror_VR(_max, عرض);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        عرض  = عرض << 1;
    }

    أرجع _max;
}

ساكن مضمن HVX_وکتور hvx_vec_reduce_max_f32(HVX_وکتور درون) {
    بیعلامت مجموع = 128;  // total vec nbytes
    بیعلامت عرض = 4;    // fp32 nbytes

    HVX_وکتور _max = درون, _max_t;
    طالما (عرض < مجموع) {
        _max_t = Q6_V_vror_VR(_max, عرض);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        عرض  = عرض << 1;
    }

    أرجع _max;
}

ساكن مضمن HVX_وکتور hvx_vec_reduce_max2_f32(HVX_وکتور درون, HVX_وکتور _max) {
    بیعلامت مجموع = 128;  // total vec nbytes
    بیعلامت عرض = 4;    // fp32 nbytes

    HVX_وکتور _max_t;

    _max = Q6_Vsf_vmax_VsfVsf(درون, _max);
    طالما (عرض < مجموع) {
        _max_t = Q6_V_vror_VR(_max, عرض);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        عرض  = عرض << 1;
    }

    أرجع _max;
}

#تعریف_کن hvx_reduce_loop_body(src_type, init_vec, pad_vec, vec_op, reduce_op, scalar_reduce) \
    افعل {                                                                                    \
        src_type * محدود vsrc = (src_type *) src;                                        \
        HVX_وکتور acc = init_vec;                                                          \
                                                                                            \
        ثابت uint32_t elem_اندازه = حجم(عائم);                                           \
        ثابت uint32_t epv  = 128 / elem_اندازه;                                              \
        ثابت uint32_t nvec = رقم_elems / epv;                                              \
        ثابت uint32_t nloe = رقم_elems % epv;                                              \
                                                                                            \
        uint32_t i = 0;                                                                     \
        _Pragma("unroll(4)")                                                                \
        لكل (; i < nvec; i++) {                                                             \
            acc = vec_op(acc, vsrc[i]);                                                     \
        }                                                                                   \
        إذا (nloe) {                                                                         \
            ثابت عائم * srcf = (ثابت عائم *) src + i * epv;                             \
            HVX_وکتور درون = *(HVX_UVector *) srcf;                                          \
            HVX_وکتور موقت = Q6_V_valign_VVR(درون, pad_vec, nloe * elem_اندازه);               \
            acc = vec_op(acc, موقت);                                                        \
        }                                                                                   \
        HVX_وکتور v = reduce_op(acc);                                                      \
        أرجع scalar_reduce(v);                                                            \
    } طالما(0)

#تعریف_کن HVX_REDUCE_MAX_OP(acc, val) Q6_Vsf_vmax_VsfVsf(acc, val)
#تعریف_کن HVX_REDUCE_جمع_OP(acc, val) Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(acc), val)
#تعریف_کن HVX_جمع_SQ_OP(acc, val) Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(val, val))
#تعریف_کن HVX_REDUCE_MAX_SCALAR(v) hvx_vec_get_f32(v)
#تعریف_کن HVX_REDUCE_جمع_SCALAR(v) hvx_vec_get_f32(Q6_Vsf_equals_Vqf32(v))

// Max variants

ساكن مضمن عائم hvx_reduce_max_f32_a(ثابت uint8_t * محدود src, ثابت صحيح رقم_elems) {
    HVX_وکتور init_vec = hvx_vec_splat_f32(((ثابت عائم *) src)[0]);
    أكد((بیعلامت دراز) src % 128 == 0);
    hvx_reduce_loop_body(HVX_وکتور, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

ساكن مضمن عائم hvx_reduce_max_f32_u(ثابت uint8_t * محدود src, ثابت صحيح رقم_elems) {
    HVX_وکتور init_vec = hvx_vec_splat_f32(((ثابت عائم *) src)[0]);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

ساكن مضمن عائم hvx_reduce_max_f32(ثابت uint8_t * محدود src, ثابت صحيح رقم_elems) {
    إذا (hex_is_aligned((فراغ *) src, 128)) {
        أرجع hvx_reduce_max_f32_a(src, رقم_elems);
    } وإلا {
        أرجع hvx_reduce_max_f32_u(src, رقم_elems);
    }
}

// Sum variants

ساكن مضمن عائم hvx_reduce_جمع_f32_a(ثابت uint8_t * محدود src, ثابت صحيح رقم_elems) {
    HVX_وکتور init_vec = Q6_V_vsplat_R(0);
    أكد((بیعلامت دراز) src % 128 == 0);
    hvx_reduce_loop_body(HVX_وکتور, init_vec, init_vec, HVX_REDUCE_جمع_OP, hvx_vec_reduce_جمع_qf32, HVX_REDUCE_جمع_SCALAR);
}

ساكن مضمن عائم hvx_reduce_جمع_f32_u(ثابت uint8_t * محدود src, ثابت صحيح رقم_elems) {
    HVX_وکتور init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_جمع_OP, hvx_vec_reduce_جمع_qf32, HVX_REDUCE_جمع_SCALAR);
}

ساكن مضمن عائم hvx_reduce_جمع_f32(ثابت uint8_t * محدود src, ثابت صحيح رقم_elems) {
    إذا (hex_is_aligned((فراغ *) src, 128)) {
        أرجع hvx_reduce_جمع_f32_a(src, رقم_elems);
    } وإلا {
        أرجع hvx_reduce_جمع_f32_u(src, رقم_elems);
    }
}

// Sum of squares variants

ساكن مضمن عائم hvx_جمع_از_squares_f32_a(ثابت uint8_t * محدود src, ثابت صحيح رقم_elems) {
    HVX_وکتور init_vec = Q6_V_vsplat_R(0);
    أكد((uintptr_t) src % 128 == 0);
    hvx_reduce_loop_body(HVX_وکتور, init_vec, init_vec, HVX_جمع_SQ_OP, hvx_vec_reduce_جمع_qf32, HVX_REDUCE_جمع_SCALAR);
}

ساكن مضمن عائم hvx_جمع_از_squares_f32_u(ثابت uint8_t * محدود src, ثابت صحيح رقم_elems) {
    HVX_وکتور init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_جمع_SQ_OP, hvx_vec_reduce_جمع_qf32, HVX_REDUCE_جمع_SCALAR);
}

ساكن مضمن عائم hvx_جمع_از_squares_f32(ثابت uint8_t * محدود src, ثابت صحيح رقم_elems) {
    إذا (hex_is_aligned((فراغ *) src, 128)) {
        أرجع hvx_جمع_از_squares_f32_a(src, رقم_elems);
    } وإلا {
        أرجع hvx_جمع_از_squares_f32_u(src, رقم_elems);
    }
}

// Signed 32-bit Integer Max variants

ساكن مضمن HVX_وکتور hvx_vec_reduce_max_n_i32(HVX_وکتور درون, بیعلامت صحيح n) {
    بیعلامت صحيح مجموع = n * 4;  // total vec nbytes
    بیعلامت صحيح عرض = 4;      // int32 nbytes

    HVX_وکتور max_val = درون, max_t;
    طالما (عرض < مجموع) {
        max_t = Q6_V_vror_VR(max_val, عرض);          // rotate right
        max_val = Q6_Vw_vmax_VwVw(max_t, max_val);      // elementwise signed max
        عرض = عرض << 1;
    }
    أرجع max_val;
}

ساكن مضمن HVX_وکتور hvx_vec_reduce_max_i32(HVX_وکتور درون) {
    أرجع hvx_vec_reduce_max_n_i32(درون, 32);
}

ساكن مضمن int32_t hvx_reduce_max_i32_a(ثابت uint8_t * محدود src, ثابت صحيح رقم_elems) {
    HVX_وکتور init_vec = Q6_V_vsplat_R(((ثابت int32_t *) src)[0]);
    HVX_وکتور pad_vec  = Q6_V_vsplat_R(0x80000000);
    أكد((uintptr_t) src % 128 == 0);
    hvx_reduce_loop_body(HVX_وکتور, init_vec, pad_vec, Q6_Vw_vmax_VwVw, hvx_vec_reduce_max_i32, hvx_vec_get_i32);
}

ساكن مضمن int32_t hvx_reduce_max_i32_u(ثابت uint8_t * محدود src, ثابت صحيح رقم_elems) {
    HVX_وکتور init_vec = Q6_V_vsplat_R(((ثابت int32_t *) src)[0]);
    HVX_وکتور pad_vec  = Q6_V_vsplat_R(0x80000000);
    hvx_reduce_loop_body(HVX_UVector, init_vec, pad_vec, Q6_Vw_vmax_VwVw, hvx_vec_reduce_max_i32, hvx_vec_get_i32);
}

ساكن مضمن int32_t hvx_reduce_max_i32(ثابت uint8_t * محدود src, ثابت صحيح رقم_elems) {
    إذا (hex_is_aligned((فراغ *) src, 128)) {
        أرجع hvx_reduce_max_i32_a(src, رقم_elems);
    } وإلا {
        أرجع hvx_reduce_max_i32_u(src, رقم_elems);
    }
}

ساكن مضمن فراغ hvx_argmax_f32(
    ثابت عائم * محدود src,
    uint32_t n,
    uint32_t جابه_جایی,
    عائم * بیرون_val,
    int32_t * بیرون_idx
) {
    إذا (n == 0) {
        *بیرون_val = -INFINITY;
        *بیرون_idx = (int32_t) جابه_جایی;
        أرجع;
    }

    إذا (n < 32 || !hex_is_aligned((فراغ *) src, 128)) {
        عائم بهترین_val = src[0];
        int32_t بهترین_idx = (int32_t) جابه_جایی;
        لكل (uint32_t i = 1; i < n; i++) {
            إذا (src[i] > بهترین_val) {
                بهترین_val = src[i];
                بهترین_idx = (int32_t) (جابه_جایی + i);
            }
        }
        *بیرون_val = بهترین_val;
        *بیرون_idx = بهترین_idx;
        أرجع;
    }

    ساكن ثابت int32_t c_lane_idx[32] __attribute__((aligned(128))) = {
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    };

    ثابت HVX_وکتور v_init_idx = *(ثابت HVX_وکتور *) c_lane_idx;
    ثابت HVX_وکتور v_گام     = Q6_V_vsplat_R(32);
    HVX_وکتور v_cur_idx        = Q6_Vw_vadd_VwVw(v_init_idx, Q6_V_vsplat_R((int32_t) جابه_جایی));
    HVX_وکتور v_max_val        = hvx_vec_splat_f32(-INFINITY);
    HVX_وکتور v_max_idx        = v_cur_idx;

    ثابت HVX_وکتور * vsrc = (ثابت HVX_وکتور *) src;
    ثابت uint32_t nvec = n / 32;

    لكل (uint32_t vi = 0; vi < nvec; vi++) {
        HVX_وکتور v = vsrc[vi];
        HVX_وکتور_Pred pred = Q6_Q_vcmp_gt_VsfVsf(v, v_max_val);
        v_max_val = Q6_V_vmux_QVV(pred, v, v_max_val);
        v_max_idx = Q6_V_vmux_QVV(pred, v_cur_idx, v_max_idx);
        v_cur_idx = Q6_Vw_vadd_VwVw(v_cur_idx, v_گام);
    }

    HVX_وکتور_Alias u_val, u_idx;
    u_val.v = v_max_val;
    u_idx.v = v_max_idx;

    عائم بهترین_val = u_val.fp32[0];
    int32_t بهترین_idx = (int32_t) u_idx.w[0];
    لكل (صحيح i = 1; i < 32; i++) {
        إذا (u_val.fp32[i] > بهترین_val) {
            بهترین_val = u_val.fp32[i];
            بهترین_idx = (int32_t) u_idx.w[i];
        }
    }

    لكل (uint32_t i = nvec * 32; i < n; i++) {
        إذا (src[i] > بهترین_val) {
            بهترین_val = src[i];
            بهترین_idx = (int32_t) (جابه_جایی + i);
        }
    }

    *بیرون_val = بهترین_val;
    *بیرون_idx = بهترین_idx;
}

#لغو_تعریف hvx_reduce_loop_body
#لغو_تعریف HVX_REDUCE_MAX_OP
#لغو_تعریف HVX_REDUCE_جمع_OP
#لغو_تعریف HVX_REDUCE_MAX_SCALAR
#لغو_تعریف HVX_REDUCE_جمع_SCALAR
#لغو_تعریف HVX_جمع_SQ_OP

#پایان_اگر /* HVX_REDUCE_H */
