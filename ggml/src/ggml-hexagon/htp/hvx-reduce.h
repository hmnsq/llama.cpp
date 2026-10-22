#اگر_غیر_تعریف_شدہ HVX_REDUCE_H
#تعریف_کرو HVX_REDUCE_H

#شامل <math.h>
#شامل <stdbool.h>
#شامل <stdint.h>
#شامل <assert.h>

#شامل "hex-utils.h"
#شامل "hvx-base.h"
#شامل "hvx-types.h"

ساکن اندرونی HVX_سمتیہ hvx_vec_reduce_مجموع_n_i32(HVX_سمتیہ اندر, غیرعلامتی صحیح n) {
    غیرعلامتی صحیح کل = n * 4;  // total vec nbytes
    غیرعلامتی صحیح چوڑائی = 4;      // int32

    HVX_سمتیہ مجموع = اندر, مجموع_t;
    جب (چوڑائی < کل) {
        مجموع_t = Q6_V_vror_VR(مجموع, چوڑائی);     // rotate right
        مجموع   = Q6_Vw_vadd_VwVw(مجموع_t, مجموع);  // elementwise sum
        چوڑائی = چوڑائی << 1;
    }
    واپس مجموع;
}

ساکن اندرونی HVX_سمتیہ hvx_vec_reduce_مجموع_i32(HVX_سمتیہ اندر) {
    واپس hvx_vec_reduce_مجموع_n_i32(اندر, 32);
}

ساکن اندرونی HVX_سمتیہ hvx_vec_reduce_مجموع_n_qf32(HVX_سمتیہ اندر, غیرعلامتی صحیح n) {
    غیرعلامتی صحیح کل = n * 4;  // total vec nbytes
    غیرعلامتی صحیح چوڑائی = 4;      // fp32 nbytes

    HVX_سمتیہ مجموع = اندر, مجموع_t;
    جب (چوڑائی < کل) {
        مجموع_t = Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(مجموع), چوڑائی);  // rotate right
        مجموع   = Q6_Vqf32_vadd_Vqf32Vsf(مجموع, مجموع_t);             // elementwise sum
        چوڑائی = چوڑائی << 1;
    }
    واپس مجموع;
}

ساکن اندرونی HVX_سمتیہ hvx_vec_reduce_مجموع_qf32(HVX_سمتیہ اندر) {
    واپس hvx_vec_reduce_مجموع_n_qf32(اندر, 32);
}

#اگر __HVX_ARCH__ > 75

ساکن اندرونی HVX_سمتیہ hvx_vec_reduce_مجموع_f32x4(HVX_سمتیہ_x4 اندر) {
    HVX_سمتیہ_Pair مجموع_p01 = Q6_W_vshuff_VVR(اندر.v[1], اندر.v[0], 4);
    HVX_سمتیہ_Pair مجموع_p23 = Q6_W_vshuff_VVR(اندر.v[3], اندر.v[2], 4);
    HVX_سمتیہ  مجموع_sf01  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(مجموع_p01), Q6_V_hi_W(مجموع_p01));
    HVX_سمتیہ  مجموع_sf23  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(مجموع_p23), Q6_V_hi_W(مجموع_p23));

    HVX_سمتیہ_Pair مجموع_p0123 = Q6_W_vshuff_VVR(مجموع_sf23, مجموع_sf01, 8);
    HVX_سمتیہ  مجموع_sf       = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(مجموع_p0123), Q6_V_hi_W(مجموع_p0123));

    مجموع_sf = Q6_Vsf_vadd_VsfVsf(مجموع_sf, Q6_V_vror_VR(مجموع_sf, VLEN / 2));
    مجموع_sf = Q6_Vsf_vadd_VsfVsf(مجموع_sf, Q6_V_vror_VR(مجموع_sf, VLEN / 4));
    مجموع_sf = Q6_Vsf_vadd_VsfVsf(مجموع_sf, Q6_V_vror_VR(مجموع_sf, VLEN / 8));
    واپس مجموع_sf;
}

ساکن اندرونی HVX_سمتیہ hvx_vec_reduce_مجموع_f32x2(HVX_سمتیہ in0, HVX_سمتیہ in1) {
    HVX_سمتیہ_Pair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_سمتیہ  مجموع_sf  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    مجموع_sf = Q6_Vsf_vadd_VsfVsf(مجموع_sf, Q6_V_vror_VR(مجموع_sf, VLEN / 2));
    مجموع_sf = Q6_Vsf_vadd_VsfVsf(مجموع_sf, Q6_V_vror_VR(مجموع_sf, VLEN / 4));
    مجموع_sf = Q6_Vsf_vadd_VsfVsf(مجموع_sf, Q6_V_vror_VR(مجموع_sf, VLEN / 8));
    مجموع_sf = Q6_Vsf_vadd_VsfVsf(مجموع_sf, Q6_V_vror_VR(مجموع_sf, VLEN / 16));
    واپس مجموع_sf;
}

ساکن اندرونی HVX_سمتیہ hvx_vec_reduce_مجموع_n_f32(HVX_سمتیہ اندر, غیرعلامتی صحیح n) {
    غیرعلامتی صحیح کل = n * 4;  // total vec nbytes
    غیرعلامتی صحیح چوڑائی = 4;      // fp32 nbytes

    HVX_سمتیہ مجموع = اندر, مجموع_t;
    جب (چوڑائی < کل) {
        مجموع_t = Q6_V_vror_VR(مجموع, چوڑائی);       // rotate right
        مجموع   = Q6_Vsf_vadd_VsfVsf(مجموع, مجموع_t); // elementwise sum
        چوڑائی = چوڑائی << 1;
    }
    واپس مجموع;
}

#ورنہ

ساکن اندرونی HVX_سمتیہ hvx_vec_reduce_مجموع_f32x4(HVX_سمتیہ_x4 اندر) {
    HVX_سمتیہ_Pair مجموع_p01  = Q6_W_vshuff_VVR(اندر.v[1], اندر.v[0], 4);
    HVX_سمتیہ_Pair مجموع_p23  = Q6_W_vshuff_VVR(اندر.v[3], اندر.v[2], 4);
    HVX_سمتیہ     مجموع_qf01 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(مجموع_p01), Q6_V_hi_W(مجموع_p01));
    HVX_سمتیہ     مجموع_qf23 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(مجموع_p23), Q6_V_hi_W(مجموع_p23));

    HVX_سمتیہ_Pair مجموع_p0123 = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(مجموع_qf23), Q6_Vsf_equals_Vqf32(مجموع_qf01), 8);
    HVX_سمتیہ     مجموع_qf    = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(مجموع_p0123), Q6_V_hi_W(مجموع_p0123));

    مجموع_qf = Q6_Vqf32_vadd_Vqf32Vsf(مجموع_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(مجموع_qf), VLEN / 2));
    مجموع_qf = Q6_Vqf32_vadd_Vqf32Vsf(مجموع_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(مجموع_qf), VLEN / 4));
    مجموع_qf = Q6_Vqf32_vadd_Vqf32Vsf(مجموع_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(مجموع_qf), VLEN / 8));
    واپس Q6_Vsf_equals_Vqf32(مجموع_qf);
}

ساکن اندرونی HVX_سمتیہ hvx_vec_reduce_مجموع_f32x2(HVX_سمتیہ in0, HVX_سمتیہ in1) {
    HVX_سمتیہ_Pair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_سمتیہ  مجموع_qf  = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    مجموع_qf = Q6_Vqf32_vadd_Vqf32Vsf(مجموع_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(مجموع_qf), VLEN / 2));
    مجموع_qf = Q6_Vqf32_vadd_Vqf32Vsf(مجموع_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(مجموع_qf), VLEN / 4));
    مجموع_qf = Q6_Vqf32_vadd_Vqf32Vsf(مجموع_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(مجموع_qf), VLEN / 8));
    مجموع_qf = Q6_Vqf32_vadd_Vqf32Vsf(مجموع_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(مجموع_qf), VLEN / 16));
    واپس Q6_Vsf_equals_Vqf32(مجموع_qf);
}

ساکن اندرونی HVX_سمتیہ hvx_vec_reduce_مجموع_n_f32(HVX_سمتیہ اندر, غیرعلامتی صحیح n) {
    غیرعلامتی صحیح کل = n * 4;  // total vec nbytes
    غیرعلامتی صحیح چوڑائی = 4;      // fp32 nbytes

    HVX_سمتیہ مجموع = اندر, مجموع_t;
    جب (چوڑائی < کل) {
        مجموع_t = Q6_V_vror_VR(مجموع, چوڑائی);                               // rotate right
        مجموع   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(مجموع, مجموع_t));  // elementwise sum
        چوڑائی = چوڑائی << 1;
    }
    واپس مجموع;
}

#اگر_ختم

ساکن اندرونی HVX_سمتیہ hvx_vec_reduce_مجموع_f32(HVX_سمتیہ اندر) {
    واپس hvx_vec_reduce_مجموع_n_f32(اندر, 32);
}

ساکن اندرونی HVX_سمتیہ hvx_vec_reduce_max_f16(HVX_سمتیہ اندر) {
    غیرعلامتی کل = 128;  // total vec nbytes
    غیرعلامتی چوڑائی = 2;    // fp16 nbytes

    HVX_سمتیہ _max = اندر, _max_t;
    جب (چوڑائی < کل) {
        _max_t = Q6_V_vror_VR(_max, چوڑائی);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        چوڑائی  = چوڑائی << 1;
    }

    واپس _max;
}

ساکن اندرونی HVX_سمتیہ hvx_vec_reduce_max2_f16(HVX_سمتیہ اندر, HVX_سمتیہ _max) {
    غیرعلامتی کل = 128;  // total vec nbytes
    غیرعلامتی چوڑائی = 2;    // fp32 nbytes

    HVX_سمتیہ _max_t;

    _max = Q6_Vhf_vmax_VhfVhf(اندر, _max);
    جب (چوڑائی < کل) {
        _max_t = Q6_V_vror_VR(_max, چوڑائی);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        چوڑائی  = چوڑائی << 1;
    }

    واپس _max;
}

ساکن اندرونی HVX_سمتیہ hvx_vec_reduce_max_f32(HVX_سمتیہ اندر) {
    غیرعلامتی کل = 128;  // total vec nbytes
    غیرعلامتی چوڑائی = 4;    // fp32 nbytes

    HVX_سمتیہ _max = اندر, _max_t;
    جب (چوڑائی < کل) {
        _max_t = Q6_V_vror_VR(_max, چوڑائی);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        چوڑائی  = چوڑائی << 1;
    }

    واپس _max;
}

ساکن اندرونی HVX_سمتیہ hvx_vec_reduce_max2_f32(HVX_سمتیہ اندر, HVX_سمتیہ _max) {
    غیرعلامتی کل = 128;  // total vec nbytes
    غیرعلامتی چوڑائی = 4;    // fp32 nbytes

    HVX_سمتیہ _max_t;

    _max = Q6_Vsf_vmax_VsfVsf(اندر, _max);
    جب (چوڑائی < کل) {
        _max_t = Q6_V_vror_VR(_max, چوڑائی);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        چوڑائی  = چوڑائی << 1;
    }

    واپس _max;
}

#تعریف_کرو hvx_reduce_loop_body(src_type, init_vec, pad_vec, vec_op, reduce_op, scalar_reduce) \
    کرو {                                                                                    \
        src_type * محدود vsrc = (src_type *) src;                                        \
        HVX_سمتیہ acc = init_vec;                                                          \
                                                                                            \
        ثابت uint32_t elem_سائز = حجم(اعشاری);                                           \
        ثابت uint32_t epv  = 128 / elem_سائز;                                              \
        ثابت uint32_t nvec = عدد_elems / epv;                                              \
        ثابت uint32_t nloe = عدد_elems % epv;                                              \
                                                                                            \
        uint32_t i = 0;                                                                     \
        _Pragma("unroll(4)")                                                                \
        برائے (; i < nvec; i++) {                                                             \
            acc = vec_op(acc, vsrc[i]);                                                     \
        }                                                                                   \
        اگر (nloe) {                                                                         \
            ثابت اعشاری * srcf = (ثابت اعشاری *) src + i * epv;                             \
            HVX_سمتیہ اندر = *(HVX_UVector *) srcf;                                          \
            HVX_سمتیہ عارضی = Q6_V_valign_VVR(اندر, pad_vec, nloe * elem_سائز);               \
            acc = vec_op(acc, عارضی);                                                        \
        }                                                                                   \
        HVX_سمتیہ v = reduce_op(acc);                                                      \
        واپس scalar_reduce(v);                                                            \
    } جب(0)

#تعریف_کرو HVX_REDUCE_MAX_OP(acc, val) Q6_Vsf_vmax_VsfVsf(acc, val)
#تعریف_کرو HVX_REDUCE_مجموع_OP(acc, val) Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(acc), val)
#تعریف_کرو HVX_مجموع_SQ_OP(acc, val) Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(val, val))
#تعریف_کرو HVX_REDUCE_MAX_SCALAR(v) hvx_vec_get_f32(v)
#تعریف_کرو HVX_REDUCE_مجموع_SCALAR(v) hvx_vec_get_f32(Q6_Vsf_equals_Vqf32(v))

// Max variants

ساکن اندرونی اعشاری hvx_reduce_max_f32_a(ثابت uint8_t * محدود src, ثابت صحیح عدد_elems) {
    HVX_سمتیہ init_vec = hvx_vec_splat_f32(((ثابت اعشاری *) src)[0]);
    أكد((غیرعلامتی طویل) src % 128 == 0);
    hvx_reduce_loop_body(HVX_سمتیہ, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

ساکن اندرونی اعشاری hvx_reduce_max_f32_u(ثابت uint8_t * محدود src, ثابت صحیح عدد_elems) {
    HVX_سمتیہ init_vec = hvx_vec_splat_f32(((ثابت اعشاری *) src)[0]);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

ساکن اندرونی اعشاری hvx_reduce_max_f32(ثابت uint8_t * محدود src, ثابت صحیح عدد_elems) {
    اگر (hex_is_aligned((خالی *) src, 128)) {
        واپس hvx_reduce_max_f32_a(src, عدد_elems);
    } ورنہ {
        واپس hvx_reduce_max_f32_u(src, عدد_elems);
    }
}

// Sum variants

ساکن اندرونی اعشاری hvx_reduce_مجموع_f32_a(ثابت uint8_t * محدود src, ثابت صحیح عدد_elems) {
    HVX_سمتیہ init_vec = Q6_V_vsplat_R(0);
    أكد((غیرعلامتی طویل) src % 128 == 0);
    hvx_reduce_loop_body(HVX_سمتیہ, init_vec, init_vec, HVX_REDUCE_مجموع_OP, hvx_vec_reduce_مجموع_qf32, HVX_REDUCE_مجموع_SCALAR);
}

ساکن اندرونی اعشاری hvx_reduce_مجموع_f32_u(ثابت uint8_t * محدود src, ثابت صحیح عدد_elems) {
    HVX_سمتیہ init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_مجموع_OP, hvx_vec_reduce_مجموع_qf32, HVX_REDUCE_مجموع_SCALAR);
}

ساکن اندرونی اعشاری hvx_reduce_مجموع_f32(ثابت uint8_t * محدود src, ثابت صحیح عدد_elems) {
    اگر (hex_is_aligned((خالی *) src, 128)) {
        واپس hvx_reduce_مجموع_f32_a(src, عدد_elems);
    } ورنہ {
        واپس hvx_reduce_مجموع_f32_u(src, عدد_elems);
    }
}

// Sum of squares variants

ساکن اندرونی اعشاری hvx_مجموع_کا_squares_f32_a(ثابت uint8_t * محدود src, ثابت صحیح عدد_elems) {
    HVX_سمتیہ init_vec = Q6_V_vsplat_R(0);
    أكد((uintptr_t) src % 128 == 0);
    hvx_reduce_loop_body(HVX_سمتیہ, init_vec, init_vec, HVX_مجموع_SQ_OP, hvx_vec_reduce_مجموع_qf32, HVX_REDUCE_مجموع_SCALAR);
}

ساکن اندرونی اعشاری hvx_مجموع_کا_squares_f32_u(ثابت uint8_t * محدود src, ثابت صحیح عدد_elems) {
    HVX_سمتیہ init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_مجموع_SQ_OP, hvx_vec_reduce_مجموع_qf32, HVX_REDUCE_مجموع_SCALAR);
}

ساکن اندرونی اعشاری hvx_مجموع_کا_squares_f32(ثابت uint8_t * محدود src, ثابت صحیح عدد_elems) {
    اگر (hex_is_aligned((خالی *) src, 128)) {
        واپس hvx_مجموع_کا_squares_f32_a(src, عدد_elems);
    } ورنہ {
        واپس hvx_مجموع_کا_squares_f32_u(src, عدد_elems);
    }
}

// Signed 32-bit Integer Max variants

ساکن اندرونی HVX_سمتیہ hvx_vec_reduce_max_n_i32(HVX_سمتیہ اندر, غیرعلامتی صحیح n) {
    غیرعلامتی صحیح کل = n * 4;  // total vec nbytes
    غیرعلامتی صحیح چوڑائی = 4;      // int32 nbytes

    HVX_سمتیہ max_val = اندر, max_t;
    جب (چوڑائی < کل) {
        max_t = Q6_V_vror_VR(max_val, چوڑائی);          // rotate right
        max_val = Q6_Vw_vmax_VwVw(max_t, max_val);      // elementwise signed max
        چوڑائی = چوڑائی << 1;
    }
    واپس max_val;
}

ساکن اندرونی HVX_سمتیہ hvx_vec_reduce_max_i32(HVX_سمتیہ اندر) {
    واپس hvx_vec_reduce_max_n_i32(اندر, 32);
}

ساکن اندرونی int32_t hvx_reduce_max_i32_a(ثابت uint8_t * محدود src, ثابت صحیح عدد_elems) {
    HVX_سمتیہ init_vec = Q6_V_vsplat_R(((ثابت int32_t *) src)[0]);
    HVX_سمتیہ pad_vec  = Q6_V_vsplat_R(0x80000000);
    أكد((uintptr_t) src % 128 == 0);
    hvx_reduce_loop_body(HVX_سمتیہ, init_vec, pad_vec, Q6_Vw_vmax_VwVw, hvx_vec_reduce_max_i32, hvx_vec_get_i32);
}

ساکن اندرونی int32_t hvx_reduce_max_i32_u(ثابت uint8_t * محدود src, ثابت صحیح عدد_elems) {
    HVX_سمتیہ init_vec = Q6_V_vsplat_R(((ثابت int32_t *) src)[0]);
    HVX_سمتیہ pad_vec  = Q6_V_vsplat_R(0x80000000);
    hvx_reduce_loop_body(HVX_UVector, init_vec, pad_vec, Q6_Vw_vmax_VwVw, hvx_vec_reduce_max_i32, hvx_vec_get_i32);
}

ساکن اندرونی int32_t hvx_reduce_max_i32(ثابت uint8_t * محدود src, ثابت صحیح عدد_elems) {
    اگر (hex_is_aligned((خالی *) src, 128)) {
        واپس hvx_reduce_max_i32_a(src, عدد_elems);
    } ورنہ {
        واپس hvx_reduce_max_i32_u(src, عدد_elems);
    }
}

ساکن اندرونی خالی hvx_argmax_f32(
    ثابت اعشاری * محدود src,
    uint32_t n,
    uint32_t انحراف,
    اعشاری * برآمد_val,
    int32_t * برآمد_idx
) {
    اگر (n == 0) {
        *برآمد_val = -INFINITY;
        *برآمد_idx = (int32_t) انحراف;
        واپس;
    }

    اگر (n < 32 || !hex_is_aligned((خالی *) src, 128)) {
        اعشاری بہترین_val = src[0];
        int32_t بہترین_idx = (int32_t) انحراف;
        برائے (uint32_t i = 1; i < n; i++) {
            اگر (src[i] > بہترین_val) {
                بہترین_val = src[i];
                بہترین_idx = (int32_t) (انحراف + i);
            }
        }
        *برآمد_val = بہترین_val;
        *برآمد_idx = بہترین_idx;
        واپس;
    }

    ساکن ثابت int32_t c_lane_idx[32] __attribute__((aligned(128))) = {
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    };

    ثابت HVX_سمتیہ v_init_idx = *(ثابت HVX_سمتیہ *) c_lane_idx;
    ثابت HVX_سمتیہ v_قدم     = Q6_V_vsplat_R(32);
    HVX_سمتیہ v_cur_idx        = Q6_Vw_vadd_VwVw(v_init_idx, Q6_V_vsplat_R((int32_t) انحراف));
    HVX_سمتیہ v_max_val        = hvx_vec_splat_f32(-INFINITY);
    HVX_سمتیہ v_max_idx        = v_cur_idx;

    ثابت HVX_سمتیہ * vsrc = (ثابت HVX_سمتیہ *) src;
    ثابت uint32_t nvec = n / 32;

    برائے (uint32_t vi = 0; vi < nvec; vi++) {
        HVX_سمتیہ v = vsrc[vi];
        HVX_سمتیہ_Pred pred = Q6_Q_vcmp_gt_VsfVsf(v, v_max_val);
        v_max_val = Q6_V_vmux_QVV(pred, v, v_max_val);
        v_max_idx = Q6_V_vmux_QVV(pred, v_cur_idx, v_max_idx);
        v_cur_idx = Q6_Vw_vadd_VwVw(v_cur_idx, v_قدم);
    }

    HVX_سمتیہ_Alias u_val, u_idx;
    u_val.v = v_max_val;
    u_idx.v = v_max_idx;

    اعشاری بہترین_val = u_val.fp32[0];
    int32_t بہترین_idx = (int32_t) u_idx.w[0];
    برائے (صحیح i = 1; i < 32; i++) {
        اگر (u_val.fp32[i] > بہترین_val) {
            بہترین_val = u_val.fp32[i];
            بہترین_idx = (int32_t) u_idx.w[i];
        }
    }

    برائے (uint32_t i = nvec * 32; i < n; i++) {
        اگر (src[i] > بہترین_val) {
            بہترین_val = src[i];
            بہترین_idx = (int32_t) (انحراف + i);
        }
    }

    *برآمد_val = بہترین_val;
    *برآمد_idx = بہترین_idx;
}

#تعریف_منسوخ hvx_reduce_loop_body
#تعریف_منسوخ HVX_REDUCE_MAX_OP
#تعریف_منسوخ HVX_REDUCE_مجموع_OP
#تعریف_منسوخ HVX_REDUCE_MAX_SCALAR
#تعریف_منسوخ HVX_REDUCE_مجموع_SCALAR
#تعریف_منسوخ HVX_مجموع_SQ_OP

#اگر_ختم /* HVX_REDUCE_H */
