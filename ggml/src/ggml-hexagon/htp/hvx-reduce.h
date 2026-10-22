#جے_بنا_تعریف HVX_REDUCE_H
#تعریف_کرو HVX_REDUCE_H

#ضمن <math.h>
#ضمن <stdbool.h>
#ضمن <stdint.h>
#ضمن <assert.h>

#ضمن "hex-utils.h"
#ضمن "hvx-base.h"
#ضمن "hvx-types.h"

ساكن مضمن HVX_سمتیہ hvx_vec_reduce_جوڑ_n_i32(HVX_سمتیہ وچ, بنانشانی صحيح n) {
    بنانشانی صحيح کل = n * 4;  // total vec nbytes
    بنانشانی صحيح چوڑائی = 4;      // int32

    HVX_سمتیہ جوڑ = وچ, جوڑ_t;
    طالما (چوڑائی < کل) {
        جوڑ_t = Q6_V_vror_VR(جوڑ, چوڑائی);     // rotate right
        جوڑ   = Q6_Vw_vadd_VwVw(جوڑ_t, جوڑ);  // elementwise sum
        چوڑائی = چوڑائی << 1;
    }
    أرجع جوڑ;
}

ساكن مضمن HVX_سمتیہ hvx_vec_reduce_جوڑ_i32(HVX_سمتیہ وچ) {
    أرجع hvx_vec_reduce_جوڑ_n_i32(وچ, 32);
}

ساكن مضمن HVX_سمتیہ hvx_vec_reduce_جوڑ_n_qf32(HVX_سمتیہ وچ, بنانشانی صحيح n) {
    بنانشانی صحيح کل = n * 4;  // total vec nbytes
    بنانشانی صحيح چوڑائی = 4;      // fp32 nbytes

    HVX_سمتیہ جوڑ = وچ, جوڑ_t;
    طالما (چوڑائی < کل) {
        جوڑ_t = Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(جوڑ), چوڑائی);  // rotate right
        جوڑ   = Q6_Vqf32_vadd_Vqf32Vsf(جوڑ, جوڑ_t);             // elementwise sum
        چوڑائی = چوڑائی << 1;
    }
    أرجع جوڑ;
}

ساكن مضمن HVX_سمتیہ hvx_vec_reduce_جوڑ_qf32(HVX_سمتیہ وچ) {
    أرجع hvx_vec_reduce_جوڑ_n_qf32(وچ, 32);
}

#إذا __HVX_ARCH__ > 75

ساكن مضمن HVX_سمتیہ hvx_vec_reduce_جوڑ_f32x4(HVX_سمتیہ_x4 وچ) {
    HVX_سمتیہ_Pair جوڑ_p01 = Q6_W_vshuff_VVR(وچ.v[1], وچ.v[0], 4);
    HVX_سمتیہ_Pair جوڑ_p23 = Q6_W_vshuff_VVR(وچ.v[3], وچ.v[2], 4);
    HVX_سمتیہ  جوڑ_sf01  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(جوڑ_p01), Q6_V_hi_W(جوڑ_p01));
    HVX_سمتیہ  جوڑ_sf23  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(جوڑ_p23), Q6_V_hi_W(جوڑ_p23));

    HVX_سمتیہ_Pair جوڑ_p0123 = Q6_W_vshuff_VVR(جوڑ_sf23, جوڑ_sf01, 8);
    HVX_سمتیہ  جوڑ_sf       = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(جوڑ_p0123), Q6_V_hi_W(جوڑ_p0123));

    جوڑ_sf = Q6_Vsf_vadd_VsfVsf(جوڑ_sf, Q6_V_vror_VR(جوڑ_sf, VLEN / 2));
    جوڑ_sf = Q6_Vsf_vadd_VsfVsf(جوڑ_sf, Q6_V_vror_VR(جوڑ_sf, VLEN / 4));
    جوڑ_sf = Q6_Vsf_vadd_VsfVsf(جوڑ_sf, Q6_V_vror_VR(جوڑ_sf, VLEN / 8));
    أرجع جوڑ_sf;
}

ساكن مضمن HVX_سمتیہ hvx_vec_reduce_جوڑ_f32x2(HVX_سمتیہ in0, HVX_سمتیہ in1) {
    HVX_سمتیہ_Pair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_سمتیہ  جوڑ_sf  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    جوڑ_sf = Q6_Vsf_vadd_VsfVsf(جوڑ_sf, Q6_V_vror_VR(جوڑ_sf, VLEN / 2));
    جوڑ_sf = Q6_Vsf_vadd_VsfVsf(جوڑ_sf, Q6_V_vror_VR(جوڑ_sf, VLEN / 4));
    جوڑ_sf = Q6_Vsf_vadd_VsfVsf(جوڑ_sf, Q6_V_vror_VR(جوڑ_sf, VLEN / 8));
    جوڑ_sf = Q6_Vsf_vadd_VsfVsf(جوڑ_sf, Q6_V_vror_VR(جوڑ_sf, VLEN / 16));
    أرجع جوڑ_sf;
}

ساكن مضمن HVX_سمتیہ hvx_vec_reduce_جوڑ_n_f32(HVX_سمتیہ وچ, بنانشانی صحيح n) {
    بنانشانی صحيح کل = n * 4;  // total vec nbytes
    بنانشانی صحيح چوڑائی = 4;      // fp32 nbytes

    HVX_سمتیہ جوڑ = وچ, جوڑ_t;
    طالما (چوڑائی < کل) {
        جوڑ_t = Q6_V_vror_VR(جوڑ, چوڑائی);       // rotate right
        جوڑ   = Q6_Vsf_vadd_VsfVsf(جوڑ, جوڑ_t); // elementwise sum
        چوڑائی = چوڑائی << 1;
    }
    أرجع جوڑ;
}

#وإلا

ساكن مضمن HVX_سمتیہ hvx_vec_reduce_جوڑ_f32x4(HVX_سمتیہ_x4 وچ) {
    HVX_سمتیہ_Pair جوڑ_p01  = Q6_W_vshuff_VVR(وچ.v[1], وچ.v[0], 4);
    HVX_سمتیہ_Pair جوڑ_p23  = Q6_W_vshuff_VVR(وچ.v[3], وچ.v[2], 4);
    HVX_سمتیہ     جوڑ_qf01 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(جوڑ_p01), Q6_V_hi_W(جوڑ_p01));
    HVX_سمتیہ     جوڑ_qf23 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(جوڑ_p23), Q6_V_hi_W(جوڑ_p23));

    HVX_سمتیہ_Pair جوڑ_p0123 = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(جوڑ_qf23), Q6_Vsf_equals_Vqf32(جوڑ_qf01), 8);
    HVX_سمتیہ     جوڑ_qf    = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(جوڑ_p0123), Q6_V_hi_W(جوڑ_p0123));

    جوڑ_qf = Q6_Vqf32_vadd_Vqf32Vsf(جوڑ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(جوڑ_qf), VLEN / 2));
    جوڑ_qf = Q6_Vqf32_vadd_Vqf32Vsf(جوڑ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(جوڑ_qf), VLEN / 4));
    جوڑ_qf = Q6_Vqf32_vadd_Vqf32Vsf(جوڑ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(جوڑ_qf), VLEN / 8));
    أرجع Q6_Vsf_equals_Vqf32(جوڑ_qf);
}

ساكن مضمن HVX_سمتیہ hvx_vec_reduce_جوڑ_f32x2(HVX_سمتیہ in0, HVX_سمتیہ in1) {
    HVX_سمتیہ_Pair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_سمتیہ  جوڑ_qf  = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    جوڑ_qf = Q6_Vqf32_vadd_Vqf32Vsf(جوڑ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(جوڑ_qf), VLEN / 2));
    جوڑ_qf = Q6_Vqf32_vadd_Vqf32Vsf(جوڑ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(جوڑ_qf), VLEN / 4));
    جوڑ_qf = Q6_Vqf32_vadd_Vqf32Vsf(جوڑ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(جوڑ_qf), VLEN / 8));
    جوڑ_qf = Q6_Vqf32_vadd_Vqf32Vsf(جوڑ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(جوڑ_qf), VLEN / 16));
    أرجع Q6_Vsf_equals_Vqf32(جوڑ_qf);
}

ساكن مضمن HVX_سمتیہ hvx_vec_reduce_جوڑ_n_f32(HVX_سمتیہ وچ, بنانشانی صحيح n) {
    بنانشانی صحيح کل = n * 4;  // total vec nbytes
    بنانشانی صحيح چوڑائی = 4;      // fp32 nbytes

    HVX_سمتیہ جوڑ = وچ, جوڑ_t;
    طالما (چوڑائی < کل) {
        جوڑ_t = Q6_V_vror_VR(جوڑ, چوڑائی);                               // rotate right
        جوڑ   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(جوڑ, جوڑ_t));  // elementwise sum
        چوڑائی = چوڑائی << 1;
    }
    أرجع جوڑ;
}

#جے_انت

ساكن مضمن HVX_سمتیہ hvx_vec_reduce_جوڑ_f32(HVX_سمتیہ وچ) {
    أرجع hvx_vec_reduce_جوڑ_n_f32(وچ, 32);
}

ساكن مضمن HVX_سمتیہ hvx_vec_reduce_max_f16(HVX_سمتیہ وچ) {
    بنانشانی کل = 128;  // total vec nbytes
    بنانشانی چوڑائی = 2;    // fp16 nbytes

    HVX_سمتیہ _max = وچ, _max_t;
    طالما (چوڑائی < کل) {
        _max_t = Q6_V_vror_VR(_max, چوڑائی);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        چوڑائی  = چوڑائی << 1;
    }

    أرجع _max;
}

ساكن مضمن HVX_سمتیہ hvx_vec_reduce_max2_f16(HVX_سمتیہ وچ, HVX_سمتیہ _max) {
    بنانشانی کل = 128;  // total vec nbytes
    بنانشانی چوڑائی = 2;    // fp32 nbytes

    HVX_سمتیہ _max_t;

    _max = Q6_Vhf_vmax_VhfVhf(وچ, _max);
    طالما (چوڑائی < کل) {
        _max_t = Q6_V_vror_VR(_max, چوڑائی);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        چوڑائی  = چوڑائی << 1;
    }

    أرجع _max;
}

ساكن مضمن HVX_سمتیہ hvx_vec_reduce_max_f32(HVX_سمتیہ وچ) {
    بنانشانی کل = 128;  // total vec nbytes
    بنانشانی چوڑائی = 4;    // fp32 nbytes

    HVX_سمتیہ _max = وچ, _max_t;
    طالما (چوڑائی < کل) {
        _max_t = Q6_V_vror_VR(_max, چوڑائی);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        چوڑائی  = چوڑائی << 1;
    }

    أرجع _max;
}

ساكن مضمن HVX_سمتیہ hvx_vec_reduce_max2_f32(HVX_سمتیہ وچ, HVX_سمتیہ _max) {
    بنانشانی کل = 128;  // total vec nbytes
    بنانشانی چوڑائی = 4;    // fp32 nbytes

    HVX_سمتیہ _max_t;

    _max = Q6_Vsf_vmax_VsfVsf(وچ, _max);
    طالما (چوڑائی < کل) {
        _max_t = Q6_V_vror_VR(_max, چوڑائی);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        چوڑائی  = چوڑائی << 1;
    }

    أرجع _max;
}

#تعریف_کرو hvx_reduce_loop_body(src_type, init_vec, pad_vec, vec_op, reduce_op, scalar_reduce) \
    افعل {                                                                                    \
        src_type * محدود vsrc = (src_type *) src;                                        \
        HVX_سمتیہ acc = init_vec;                                                          \
                                                                                            \
        ثابت uint32_t elem_ناپ = حجم(عائم);                                           \
        ثابت uint32_t epv  = 128 / elem_ناپ;                                              \
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
            HVX_سمتیہ وچ = *(HVX_UVector *) srcf;                                          \
            HVX_سمتیہ عارضی = Q6_V_valign_VVR(وچ, pad_vec, nloe * elem_ناپ);               \
            acc = vec_op(acc, عارضی);                                                        \
        }                                                                                   \
        HVX_سمتیہ v = reduce_op(acc);                                                      \
        أرجع scalar_reduce(v);                                                            \
    } طالما(0)

#تعریف_کرو HVX_REDUCE_MAX_OP(acc, val) Q6_Vsf_vmax_VsfVsf(acc, val)
#تعریف_کرو HVX_REDUCE_جوڑ_OP(acc, val) Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(acc), val)
#تعریف_کرو HVX_جوڑ_SQ_OP(acc, val) Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(val, val))
#تعریف_کرو HVX_REDUCE_MAX_SCALAR(v) hvx_vec_get_f32(v)
#تعریف_کرو HVX_REDUCE_جوڑ_SCALAR(v) hvx_vec_get_f32(Q6_Vsf_equals_Vqf32(v))

// Max variants

ساكن مضمن عائم hvx_reduce_max_f32_a(ثابت uint8_t * محدود src, ثابت صحيح عدد_elems) {
    HVX_سمتیہ init_vec = hvx_vec_splat_f32(((ثابت عائم *) src)[0]);
    أكد((بنانشانی لما) src % 128 == 0);
    hvx_reduce_loop_body(HVX_سمتیہ, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

ساكن مضمن عائم hvx_reduce_max_f32_u(ثابت uint8_t * محدود src, ثابت صحيح عدد_elems) {
    HVX_سمتیہ init_vec = hvx_vec_splat_f32(((ثابت عائم *) src)[0]);
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

ساكن مضمن عائم hvx_reduce_جوڑ_f32_a(ثابت uint8_t * محدود src, ثابت صحيح عدد_elems) {
    HVX_سمتیہ init_vec = Q6_V_vsplat_R(0);
    أكد((بنانشانی لما) src % 128 == 0);
    hvx_reduce_loop_body(HVX_سمتیہ, init_vec, init_vec, HVX_REDUCE_جوڑ_OP, hvx_vec_reduce_جوڑ_qf32, HVX_REDUCE_جوڑ_SCALAR);
}

ساكن مضمن عائم hvx_reduce_جوڑ_f32_u(ثابت uint8_t * محدود src, ثابت صحيح عدد_elems) {
    HVX_سمتیہ init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_جوڑ_OP, hvx_vec_reduce_جوڑ_qf32, HVX_REDUCE_جوڑ_SCALAR);
}

ساكن مضمن عائم hvx_reduce_جوڑ_f32(ثابت uint8_t * محدود src, ثابت صحيح عدد_elems) {
    إذا (hex_is_aligned((فراغ *) src, 128)) {
        أرجع hvx_reduce_جوڑ_f32_a(src, عدد_elems);
    } وإلا {
        أرجع hvx_reduce_جوڑ_f32_u(src, عدد_elems);
    }
}

// Sum of squares variants

ساكن مضمن عائم hvx_جوڑ_دا_squares_f32_a(ثابت uint8_t * محدود src, ثابت صحيح عدد_elems) {
    HVX_سمتیہ init_vec = Q6_V_vsplat_R(0);
    أكد((uintptr_t) src % 128 == 0);
    hvx_reduce_loop_body(HVX_سمتیہ, init_vec, init_vec, HVX_جوڑ_SQ_OP, hvx_vec_reduce_جوڑ_qf32, HVX_REDUCE_جوڑ_SCALAR);
}

ساكن مضمن عائم hvx_جوڑ_دا_squares_f32_u(ثابت uint8_t * محدود src, ثابت صحيح عدد_elems) {
    HVX_سمتیہ init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_جوڑ_SQ_OP, hvx_vec_reduce_جوڑ_qf32, HVX_REDUCE_جوڑ_SCALAR);
}

ساكن مضمن عائم hvx_جوڑ_دا_squares_f32(ثابت uint8_t * محدود src, ثابت صحيح عدد_elems) {
    إذا (hex_is_aligned((فراغ *) src, 128)) {
        أرجع hvx_جوڑ_دا_squares_f32_a(src, عدد_elems);
    } وإلا {
        أرجع hvx_جوڑ_دا_squares_f32_u(src, عدد_elems);
    }
}

// Signed 32-bit Integer Max variants

ساكن مضمن HVX_سمتیہ hvx_vec_reduce_max_n_i32(HVX_سمتیہ وچ, بنانشانی صحيح n) {
    بنانشانی صحيح کل = n * 4;  // total vec nbytes
    بنانشانی صحيح چوڑائی = 4;      // int32 nbytes

    HVX_سمتیہ max_val = وچ, max_t;
    طالما (چوڑائی < کل) {
        max_t = Q6_V_vror_VR(max_val, چوڑائی);          // rotate right
        max_val = Q6_Vw_vmax_VwVw(max_t, max_val);      // elementwise signed max
        چوڑائی = چوڑائی << 1;
    }
    أرجع max_val;
}

ساكن مضمن HVX_سمتیہ hvx_vec_reduce_max_i32(HVX_سمتیہ وچ) {
    أرجع hvx_vec_reduce_max_n_i32(وچ, 32);
}

ساكن مضمن int32_t hvx_reduce_max_i32_a(ثابت uint8_t * محدود src, ثابت صحيح عدد_elems) {
    HVX_سمتیہ init_vec = Q6_V_vsplat_R(((ثابت int32_t *) src)[0]);
    HVX_سمتیہ pad_vec  = Q6_V_vsplat_R(0x80000000);
    أكد((uintptr_t) src % 128 == 0);
    hvx_reduce_loop_body(HVX_سمتیہ, init_vec, pad_vec, Q6_Vw_vmax_VwVw, hvx_vec_reduce_max_i32, hvx_vec_get_i32);
}

ساكن مضمن int32_t hvx_reduce_max_i32_u(ثابت uint8_t * محدود src, ثابت صحيح عدد_elems) {
    HVX_سمتیہ init_vec = Q6_V_vsplat_R(((ثابت int32_t *) src)[0]);
    HVX_سمتیہ pad_vec  = Q6_V_vsplat_R(0x80000000);
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
    uint32_t وتھ,
    عائم * باہرلا_val,
    int32_t * باہرلا_idx
) {
    إذا (n == 0) {
        *باہرلا_val = -INFINITY;
        *باہرلا_idx = (int32_t) وتھ;
        أرجع;
    }

    إذا (n < 32 || !hex_is_aligned((فراغ *) src, 128)) {
        عائم بہترین_val = src[0];
        int32_t بہترین_idx = (int32_t) وتھ;
        لكل (uint32_t i = 1; i < n; i++) {
            إذا (src[i] > بہترین_val) {
                بہترین_val = src[i];
                بہترین_idx = (int32_t) (وتھ + i);
            }
        }
        *باہرلا_val = بہترین_val;
        *باہرلا_idx = بہترین_idx;
        أرجع;
    }

    ساكن ثابت int32_t c_lane_idx[32] __attribute__((aligned(128))) = {
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    };

    ثابت HVX_سمتیہ v_init_idx = *(ثابت HVX_سمتیہ *) c_lane_idx;
    ثابت HVX_سمتیہ v_قدم     = Q6_V_vsplat_R(32);
    HVX_سمتیہ v_cur_idx        = Q6_Vw_vadd_VwVw(v_init_idx, Q6_V_vsplat_R((int32_t) وتھ));
    HVX_سمتیہ v_max_val        = hvx_vec_splat_f32(-INFINITY);
    HVX_سمتیہ v_max_idx        = v_cur_idx;

    ثابت HVX_سمتیہ * vsrc = (ثابت HVX_سمتیہ *) src;
    ثابت uint32_t nvec = n / 32;

    لكل (uint32_t vi = 0; vi < nvec; vi++) {
        HVX_سمتیہ v = vsrc[vi];
        HVX_سمتیہ_Pred pred = Q6_Q_vcmp_gt_VsfVsf(v, v_max_val);
        v_max_val = Q6_V_vmux_QVV(pred, v, v_max_val);
        v_max_idx = Q6_V_vmux_QVV(pred, v_cur_idx, v_max_idx);
        v_cur_idx = Q6_Vw_vadd_VwVw(v_cur_idx, v_قدم);
    }

    HVX_سمتیہ_Alias u_val, u_idx;
    u_val.v = v_max_val;
    u_idx.v = v_max_idx;

    عائم بہترین_val = u_val.fp32[0];
    int32_t بہترین_idx = (int32_t) u_idx.w[0];
    لكل (صحيح i = 1; i < 32; i++) {
        إذا (u_val.fp32[i] > بہترین_val) {
            بہترین_val = u_val.fp32[i];
            بہترین_idx = (int32_t) u_idx.w[i];
        }
    }

    لكل (uint32_t i = nvec * 32; i < n; i++) {
        إذا (src[i] > بہترین_val) {
            بہترین_val = src[i];
            بہترین_idx = (int32_t) (وتھ + i);
        }
    }

    *باہرلا_val = بہترین_val;
    *باہرلا_idx = بہترین_idx;
}

#تعریف_رد hvx_reduce_loop_body
#تعریف_رد HVX_REDUCE_MAX_OP
#تعریف_رد HVX_REDUCE_جوڑ_OP
#تعریف_رد HVX_REDUCE_MAX_SCALAR
#تعریف_رد HVX_REDUCE_جوڑ_SCALAR
#تعریف_رد HVX_جوڑ_SQ_OP

#جے_انت /* HVX_REDUCE_H */
