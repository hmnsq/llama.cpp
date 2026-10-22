#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HVX_REDUCE_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_REDUCE_H

#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-base.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-types.h"

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_reduce_ਜੋੜ_n_i32(HVX_ਸਦਿਸ਼ ਵਿੱਚ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ਕੁੱਲ = n * 4;  // total vec nbytes
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ਚੌੜਾਈ = 4;      // int32

    HVX_ਸਦਿਸ਼ ਜੋੜ = ਵਿੱਚ, ਜੋੜ_t;
    ਜਦੋਂ (ਚੌੜਾਈ < ਕੁੱਲ) {
        ਜੋੜ_t = Q6_V_vror_VR(ਜੋੜ, ਚੌੜਾਈ);     // rotate right
        ਜੋੜ   = Q6_Vw_vadd_VwVw(ਜੋੜ_t, ਜੋੜ);  // elementwise sum
        ਚੌੜਾਈ = ਚੌੜਾਈ << 1;
    }
    ਵਾਪਸ ਜੋੜ;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_reduce_ਜੋੜ_i32(HVX_ਸਦਿਸ਼ ਵਿੱਚ) {
    ਵਾਪਸ hvx_vec_reduce_ਜੋੜ_n_i32(ਵਿੱਚ, 32);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_reduce_ਜੋੜ_n_qf32(HVX_ਸਦਿਸ਼ ਵਿੱਚ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ਕੁੱਲ = n * 4;  // total vec nbytes
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ਚੌੜਾਈ = 4;      // fp32 nbytes

    HVX_ਸਦਿਸ਼ ਜੋੜ = ਵਿੱਚ, ਜੋੜ_t;
    ਜਦੋਂ (ਚੌੜਾਈ < ਕੁੱਲ) {
        ਜੋੜ_t = Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(ਜੋੜ), ਚੌੜਾਈ);  // rotate right
        ਜੋੜ   = Q6_Vqf32_vadd_Vqf32Vsf(ਜੋੜ, ਜੋੜ_t);             // elementwise sum
        ਚੌੜਾਈ = ਚੌੜਾਈ << 1;
    }
    ਵਾਪਸ ਜੋੜ;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_reduce_ਜੋੜ_qf32(HVX_ਸਦਿਸ਼ ਵਿੱਚ) {
    ਵਾਪਸ hvx_vec_reduce_ਜੋੜ_n_qf32(ਵਿੱਚ, 32);
}

#ਜੇ __HVX_ARCH__ > 75

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_reduce_ਜੋੜ_f32x4(HVX_ਸਦਿਸ਼_x4 ਵਿੱਚ) {
    HVX_ਸਦਿਸ਼_Pair ਜੋੜ_p01 = Q6_W_vshuff_VVR(ਵਿੱਚ.v[1], ਵਿੱਚ.v[0], 4);
    HVX_ਸਦਿਸ਼_Pair ਜੋੜ_p23 = Q6_W_vshuff_VVR(ਵਿੱਚ.v[3], ਵਿੱਚ.v[2], 4);
    HVX_ਸਦਿਸ਼  ਜੋੜ_sf01  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(ਜੋੜ_p01), Q6_V_hi_W(ਜੋੜ_p01));
    HVX_ਸਦਿਸ਼  ਜੋੜ_sf23  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(ਜੋੜ_p23), Q6_V_hi_W(ਜੋੜ_p23));

    HVX_ਸਦਿਸ਼_Pair ਜੋੜ_p0123 = Q6_W_vshuff_VVR(ਜੋੜ_sf23, ਜੋੜ_sf01, 8);
    HVX_ਸਦਿਸ਼  ਜੋੜ_sf       = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(ਜੋੜ_p0123), Q6_V_hi_W(ਜੋੜ_p0123));

    ਜੋੜ_sf = Q6_Vsf_vadd_VsfVsf(ਜੋੜ_sf, Q6_V_vror_VR(ਜੋੜ_sf, VLEN / 2));
    ਜੋੜ_sf = Q6_Vsf_vadd_VsfVsf(ਜੋੜ_sf, Q6_V_vror_VR(ਜੋੜ_sf, VLEN / 4));
    ਜੋੜ_sf = Q6_Vsf_vadd_VsfVsf(ਜੋੜ_sf, Q6_V_vror_VR(ਜੋੜ_sf, VLEN / 8));
    ਵਾਪਸ ਜੋੜ_sf;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_reduce_ਜੋੜ_f32x2(HVX_ਸਦਿਸ਼ in0, HVX_ਸਦਿਸ਼ in1) {
    HVX_ਸਦਿਸ਼_Pair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_ਸਦਿਸ਼  ਜੋੜ_sf  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    ਜੋੜ_sf = Q6_Vsf_vadd_VsfVsf(ਜੋੜ_sf, Q6_V_vror_VR(ਜੋੜ_sf, VLEN / 2));
    ਜੋੜ_sf = Q6_Vsf_vadd_VsfVsf(ਜੋੜ_sf, Q6_V_vror_VR(ਜੋੜ_sf, VLEN / 4));
    ਜੋੜ_sf = Q6_Vsf_vadd_VsfVsf(ਜੋੜ_sf, Q6_V_vror_VR(ਜੋੜ_sf, VLEN / 8));
    ਜੋੜ_sf = Q6_Vsf_vadd_VsfVsf(ਜੋੜ_sf, Q6_V_vror_VR(ਜੋੜ_sf, VLEN / 16));
    ਵਾਪਸ ਜੋੜ_sf;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_reduce_ਜੋੜ_n_f32(HVX_ਸਦਿਸ਼ ਵਿੱਚ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ਕੁੱਲ = n * 4;  // total vec nbytes
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ਚੌੜਾਈ = 4;      // fp32 nbytes

    HVX_ਸਦਿਸ਼ ਜੋੜ = ਵਿੱਚ, ਜੋੜ_t;
    ਜਦੋਂ (ਚੌੜਾਈ < ਕੁੱਲ) {
        ਜੋੜ_t = Q6_V_vror_VR(ਜੋੜ, ਚੌੜਾਈ);       // rotate right
        ਜੋੜ   = Q6_Vsf_vadd_VsfVsf(ਜੋੜ, ਜੋੜ_t); // elementwise sum
        ਚੌੜਾਈ = ਚੌੜਾਈ << 1;
    }
    ਵਾਪਸ ਜੋੜ;
}

#ਨਹੀਂਤਾਂ

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_reduce_ਜੋੜ_f32x4(HVX_ਸਦਿਸ਼_x4 ਵਿੱਚ) {
    HVX_ਸਦਿਸ਼_Pair ਜੋੜ_p01  = Q6_W_vshuff_VVR(ਵਿੱਚ.v[1], ਵਿੱਚ.v[0], 4);
    HVX_ਸਦਿਸ਼_Pair ਜੋੜ_p23  = Q6_W_vshuff_VVR(ਵਿੱਚ.v[3], ਵਿੱਚ.v[2], 4);
    HVX_ਸਦਿਸ਼     ਜੋੜ_qf01 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(ਜੋੜ_p01), Q6_V_hi_W(ਜੋੜ_p01));
    HVX_ਸਦਿਸ਼     ਜੋੜ_qf23 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(ਜੋੜ_p23), Q6_V_hi_W(ਜੋੜ_p23));

    HVX_ਸਦਿਸ਼_Pair ਜੋੜ_p0123 = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(ਜੋੜ_qf23), Q6_Vsf_equals_Vqf32(ਜੋੜ_qf01), 8);
    HVX_ਸਦਿਸ਼     ਜੋੜ_qf    = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(ਜੋੜ_p0123), Q6_V_hi_W(ਜੋੜ_p0123));

    ਜੋੜ_qf = Q6_Vqf32_vadd_Vqf32Vsf(ਜੋੜ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(ਜੋੜ_qf), VLEN / 2));
    ਜੋੜ_qf = Q6_Vqf32_vadd_Vqf32Vsf(ਜੋੜ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(ਜੋੜ_qf), VLEN / 4));
    ਜੋੜ_qf = Q6_Vqf32_vadd_Vqf32Vsf(ਜੋੜ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(ਜੋੜ_qf), VLEN / 8));
    ਵਾਪਸ Q6_Vsf_equals_Vqf32(ਜੋੜ_qf);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_reduce_ਜੋੜ_f32x2(HVX_ਸਦਿਸ਼ in0, HVX_ਸਦਿਸ਼ in1) {
    HVX_ਸਦਿਸ਼_Pair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_ਸਦਿਸ਼  ਜੋੜ_qf  = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    ਜੋੜ_qf = Q6_Vqf32_vadd_Vqf32Vsf(ਜੋੜ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(ਜੋੜ_qf), VLEN / 2));
    ਜੋੜ_qf = Q6_Vqf32_vadd_Vqf32Vsf(ਜੋੜ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(ਜੋੜ_qf), VLEN / 4));
    ਜੋੜ_qf = Q6_Vqf32_vadd_Vqf32Vsf(ਜੋੜ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(ਜੋੜ_qf), VLEN / 8));
    ਜੋੜ_qf = Q6_Vqf32_vadd_Vqf32Vsf(ਜੋੜ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(ਜੋੜ_qf), VLEN / 16));
    ਵਾਪਸ Q6_Vsf_equals_Vqf32(ਜੋੜ_qf);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_reduce_ਜੋੜ_n_f32(HVX_ਸਦਿਸ਼ ਵਿੱਚ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ਕੁੱਲ = n * 4;  // total vec nbytes
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ਚੌੜਾਈ = 4;      // fp32 nbytes

    HVX_ਸਦਿਸ਼ ਜੋੜ = ਵਿੱਚ, ਜੋੜ_t;
    ਜਦੋਂ (ਚੌੜਾਈ < ਕੁੱਲ) {
        ਜੋੜ_t = Q6_V_vror_VR(ਜੋੜ, ਚੌੜਾਈ);                               // rotate right
        ਜੋੜ   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(ਜੋੜ, ਜੋੜ_t));  // elementwise sum
        ਚੌੜਾਈ = ਚੌੜਾਈ << 1;
    }
    ਵਾਪਸ ਜੋੜ;
}

#ਜੇ_ਅੰਤ

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_reduce_ਜੋੜ_f32(HVX_ਸਦਿਸ਼ ਵਿੱਚ) {
    ਵਾਪਸ hvx_vec_reduce_ਜੋੜ_n_f32(ਵਿੱਚ, 32);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_reduce_max_f16(HVX_ਸਦਿਸ਼ ਵਿੱਚ) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਕੁੱਲ = 128;  // total vec nbytes
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਚੌੜਾਈ = 2;    // fp16 nbytes

    HVX_ਸਦਿਸ਼ _max = ਵਿੱਚ, _max_t;
    ਜਦੋਂ (ਚੌੜਾਈ < ਕੁੱਲ) {
        _max_t = Q6_V_vror_VR(_max, ਚੌੜਾਈ);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        ਚੌੜਾਈ  = ਚੌੜਾਈ << 1;
    }

    ਵਾਪਸ _max;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_reduce_max2_f16(HVX_ਸਦਿਸ਼ ਵਿੱਚ, HVX_ਸਦਿਸ਼ _max) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਕੁੱਲ = 128;  // total vec nbytes
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਚੌੜਾਈ = 2;    // fp32 nbytes

    HVX_ਸਦਿਸ਼ _max_t;

    _max = Q6_Vhf_vmax_VhfVhf(ਵਿੱਚ, _max);
    ਜਦੋਂ (ਚੌੜਾਈ < ਕੁੱਲ) {
        _max_t = Q6_V_vror_VR(_max, ਚੌੜਾਈ);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        ਚੌੜਾਈ  = ਚੌੜਾਈ << 1;
    }

    ਵਾਪਸ _max;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_reduce_max_f32(HVX_ਸਦਿਸ਼ ਵਿੱਚ) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਕੁੱਲ = 128;  // total vec nbytes
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਚੌੜਾਈ = 4;    // fp32 nbytes

    HVX_ਸਦਿਸ਼ _max = ਵਿੱਚ, _max_t;
    ਜਦੋਂ (ਚੌੜਾਈ < ਕੁੱਲ) {
        _max_t = Q6_V_vror_VR(_max, ਚੌੜਾਈ);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        ਚੌੜਾਈ  = ਚੌੜਾਈ << 1;
    }

    ਵਾਪਸ _max;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_reduce_max2_f32(HVX_ਸਦਿਸ਼ ਵਿੱਚ, HVX_ਸਦਿਸ਼ _max) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਕੁੱਲ = 128;  // total vec nbytes
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਚੌੜਾਈ = 4;    // fp32 nbytes

    HVX_ਸਦਿਸ਼ _max_t;

    _max = Q6_Vsf_vmax_VsfVsf(ਵਿੱਚ, _max);
    ਜਦੋਂ (ਚੌੜਾਈ < ਕੁੱਲ) {
        _max_t = Q6_V_vror_VR(_max, ਚੌੜਾਈ);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        ਚੌੜਾਈ  = ਚੌੜਾਈ << 1;
    }

    ਵਾਪਸ _max;
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ hvx_reduce_loop_body(src_type, init_vec, pad_vec, vec_op, reduce_op, scalar_reduce) \
    ਕਰੋ {                                                                                    \
        src_type * ਸੀਮਿਤ vsrc = (src_type *) src;                                        \
        HVX_ਸਦਿਸ਼ acc = init_vec;                                                          \
                                                                                            \
        ਅਚਲ uint32_t elem_ਮਾਪ = ਆਕਾਰ(ਦਸ਼ਮਲਵ);                                           \
        ਅਚਲ uint32_t epv  = 128 / elem_ਮਾਪ;                                              \
        ਅਚਲ uint32_t nvec = ਅੰਕ_elems / epv;                                              \
        ਅਚਲ uint32_t nloe = ਅੰਕ_elems % epv;                                              \
                                                                                            \
        uint32_t i = 0;                                                                     \
        _Pragma("unroll(4)")                                                                \
        ਲਈ (; i < nvec; i++) {                                                             \
            acc = vec_op(acc, vsrc[i]);                                                     \
        }                                                                                   \
        ਜੇ (nloe) {                                                                         \
            ਅਚਲ ਦਸ਼ਮਲਵ * srcf = (ਅਚਲ ਦਸ਼ਮਲਵ *) src + i * epv;                             \
            HVX_ਸਦਿਸ਼ ਵਿੱਚ = *(HVX_UVector *) srcf;                                          \
            HVX_ਸਦਿਸ਼ ਆਰਜ਼ੀ = Q6_V_valign_VVR(ਵਿੱਚ, pad_vec, nloe * elem_ਮਾਪ);               \
            acc = vec_op(acc, ਆਰਜ਼ੀ);                                                        \
        }                                                                                   \
        HVX_ਸਦਿਸ਼ v = reduce_op(acc);                                                      \
        ਵਾਪਸ scalar_reduce(v);                                                            \
    } ਜਦੋਂ(0)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_REDUCE_MAX_OP(acc, val) Q6_Vsf_vmax_VsfVsf(acc, val)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_REDUCE_ਜੋੜ_OP(acc, val) Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(acc), val)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_ਜੋੜ_SQ_OP(acc, val) Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(val, val))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_REDUCE_MAX_SCALAR(v) hvx_vec_get_f32(v)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_REDUCE_ਜੋੜ_SCALAR(v) hvx_vec_get_f32(Q6_Vsf_equals_Vqf32(v))

// Max variants

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ hvx_reduce_max_f32_a(ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ ਅੰਕ_elems) {
    HVX_ਸਦਿਸ਼ init_vec = hvx_vec_splat_f32(((ਅਚਲ ਦਸ਼ਮਲਵ *) src)[0]);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_reduce_loop_body(HVX_ਸਦਿਸ਼, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ hvx_reduce_max_f32_u(ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ ਅੰਕ_elems) {
    HVX_ਸਦਿਸ਼ init_vec = hvx_vec_splat_f32(((ਅਚਲ ਦਸ਼ਮਲਵ *) src)[0]);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ hvx_reduce_max_f32(ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ ਅੰਕ_elems) {
    ਜੇ (hex_is_aligned((ਖਾਲੀ *) src, 128)) {
        ਵਾਪਸ hvx_reduce_max_f32_a(src, ਅੰਕ_elems);
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ hvx_reduce_max_f32_u(src, ਅੰਕ_elems);
    }
}

// Sum variants

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ hvx_reduce_ਜੋੜ_f32_a(ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ ਅੰਕ_elems) {
    HVX_ਸਦਿਸ਼ init_vec = Q6_V_vsplat_R(0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_reduce_loop_body(HVX_ਸਦਿਸ਼, init_vec, init_vec, HVX_REDUCE_ਜੋੜ_OP, hvx_vec_reduce_ਜੋੜ_qf32, HVX_REDUCE_ਜੋੜ_SCALAR);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ hvx_reduce_ਜੋੜ_f32_u(ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ ਅੰਕ_elems) {
    HVX_ਸਦਿਸ਼ init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_ਜੋੜ_OP, hvx_vec_reduce_ਜੋੜ_qf32, HVX_REDUCE_ਜੋੜ_SCALAR);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ hvx_reduce_ਜੋੜ_f32(ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ ਅੰਕ_elems) {
    ਜੇ (hex_is_aligned((ਖਾਲੀ *) src, 128)) {
        ਵਾਪਸ hvx_reduce_ਜੋੜ_f32_a(src, ਅੰਕ_elems);
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ hvx_reduce_ਜੋੜ_f32_u(src, ਅੰਕ_elems);
    }
}

// Sum of squares variants

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ hvx_ਜੋੜ_ਦਾ_squares_f32_a(ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ ਅੰਕ_elems) {
    HVX_ਸਦਿਸ਼ init_vec = Q6_V_vsplat_R(0);
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) src % 128 == 0);
    hvx_reduce_loop_body(HVX_ਸਦਿਸ਼, init_vec, init_vec, HVX_ਜੋੜ_SQ_OP, hvx_vec_reduce_ਜੋੜ_qf32, HVX_REDUCE_ਜੋੜ_SCALAR);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ hvx_ਜੋੜ_ਦਾ_squares_f32_u(ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ ਅੰਕ_elems) {
    HVX_ਸਦਿਸ਼ init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_ਜੋੜ_SQ_OP, hvx_vec_reduce_ਜੋੜ_qf32, HVX_REDUCE_ਜੋੜ_SCALAR);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ hvx_ਜੋੜ_ਦਾ_squares_f32(ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ ਅੰਕ_elems) {
    ਜੇ (hex_is_aligned((ਖਾਲੀ *) src, 128)) {
        ਵਾਪਸ hvx_ਜੋੜ_ਦਾ_squares_f32_a(src, ਅੰਕ_elems);
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ hvx_ਜੋੜ_ਦਾ_squares_f32_u(src, ਅੰਕ_elems);
    }
}

// Signed 32-bit Integer Max variants

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_reduce_max_n_i32(HVX_ਸਦਿਸ਼ ਵਿੱਚ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ਕੁੱਲ = n * 4;  // total vec nbytes
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ਚੌੜਾਈ = 4;      // int32 nbytes

    HVX_ਸਦਿਸ਼ max_val = ਵਿੱਚ, max_t;
    ਜਦੋਂ (ਚੌੜਾਈ < ਕੁੱਲ) {
        max_t = Q6_V_vror_VR(max_val, ਚੌੜਾਈ);          // rotate right
        max_val = Q6_Vw_vmax_VwVw(max_t, max_val);      // elementwise signed max
        ਚੌੜਾਈ = ਚੌੜਾਈ << 1;
    }
    ਵਾਪਸ max_val;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_reduce_max_i32(HVX_ਸਦਿਸ਼ ਵਿੱਚ) {
    ਵਾਪਸ hvx_vec_reduce_max_n_i32(ਵਿੱਚ, 32);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ int32_t hvx_reduce_max_i32_a(ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ ਅੰਕ_elems) {
    HVX_ਸਦਿਸ਼ init_vec = Q6_V_vsplat_R(((ਅਚਲ int32_t *) src)[0]);
    HVX_ਸਦਿਸ਼ pad_vec  = Q6_V_vsplat_R(0x80000000);
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) src % 128 == 0);
    hvx_reduce_loop_body(HVX_ਸਦਿਸ਼, init_vec, pad_vec, Q6_Vw_vmax_VwVw, hvx_vec_reduce_max_i32, hvx_vec_get_i32);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ int32_t hvx_reduce_max_i32_u(ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ ਅੰਕ_elems) {
    HVX_ਸਦਿਸ਼ init_vec = Q6_V_vsplat_R(((ਅਚਲ int32_t *) src)[0]);
    HVX_ਸਦਿਸ਼ pad_vec  = Q6_V_vsplat_R(0x80000000);
    hvx_reduce_loop_body(HVX_UVector, init_vec, pad_vec, Q6_Vw_vmax_VwVw, hvx_vec_reduce_max_i32, hvx_vec_get_i32);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ int32_t hvx_reduce_max_i32(ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ ਅੰਕ_elems) {
    ਜੇ (hex_is_aligned((ਖਾਲੀ *) src, 128)) {
        ਵਾਪਸ hvx_reduce_max_i32_a(src, ਅੰਕ_elems);
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ hvx_reduce_max_i32_u(src, ਅੰਕ_elems);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_argmax_f32(
    ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src,
    uint32_t n,
    uint32_t ਵਿੱਥ,
    ਦਸ਼ਮਲਵ * ਬਾਹਰਲਾ_val,
    int32_t * ਬਾਹਰਲਾ_idx
) {
    ਜੇ (n == 0) {
        *ਬਾਹਰਲਾ_val = -INFINITY;
        *ਬਾਹਰਲਾ_idx = (int32_t) ਵਿੱਥ;
        ਵਾਪਸ;
    }

    ਜੇ (n < 32 || !hex_is_aligned((ਖਾਲੀ *) src, 128)) {
        ਦਸ਼ਮਲਵ ਉੱਤਮ_val = src[0];
        int32_t ਉੱਤਮ_idx = (int32_t) ਵਿੱਥ;
        ਲਈ (uint32_t i = 1; i < n; i++) {
            ਜੇ (src[i] > ਉੱਤਮ_val) {
                ਉੱਤਮ_val = src[i];
                ਉੱਤਮ_idx = (int32_t) (ਵਿੱਥ + i);
            }
        }
        *ਬਾਹਰਲਾ_val = ਉੱਤਮ_val;
        *ਬਾਹਰਲਾ_idx = ਉੱਤਮ_idx;
        ਵਾਪਸ;
    }

    ਸਥਿਰ ਅਚਲ int32_t c_lane_idx[32] __attribute__((aligned(128))) = {
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    };

    ਅਚਲ HVX_ਸਦਿਸ਼ v_init_idx = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) c_lane_idx;
    ਅਚਲ HVX_ਸਦਿਸ਼ v_ਕਦਮ     = Q6_V_vsplat_R(32);
    HVX_ਸਦਿਸ਼ v_cur_idx        = Q6_Vw_vadd_VwVw(v_init_idx, Q6_V_vsplat_R((int32_t) ਵਿੱਥ));
    HVX_ਸਦਿਸ਼ v_max_val        = hvx_vec_splat_f32(-INFINITY);
    HVX_ਸਦਿਸ਼ v_max_idx        = v_cur_idx;

    ਅਚਲ HVX_ਸਦਿਸ਼ * vsrc = (ਅਚਲ HVX_ਸਦਿਸ਼ *) src;
    ਅਚਲ uint32_t nvec = n / 32;

    ਲਈ (uint32_t vi = 0; vi < nvec; vi++) {
        HVX_ਸਦਿਸ਼ v = vsrc[vi];
        HVX_ਸਦਿਸ਼_Pred pred = Q6_Q_vcmp_gt_VsfVsf(v, v_max_val);
        v_max_val = Q6_V_vmux_QVV(pred, v, v_max_val);
        v_max_idx = Q6_V_vmux_QVV(pred, v_cur_idx, v_max_idx);
        v_cur_idx = Q6_Vw_vadd_VwVw(v_cur_idx, v_ਕਦਮ);
    }

    HVX_ਸਦਿਸ਼_Alias u_val, u_idx;
    u_val.v = v_max_val;
    u_idx.v = v_max_idx;

    ਦਸ਼ਮਲਵ ਉੱਤਮ_val = u_val.fp32[0];
    int32_t ਉੱਤਮ_idx = (int32_t) u_idx.w[0];
    ਲਈ (ਪੂਰਨ i = 1; i < 32; i++) {
        ਜੇ (u_val.fp32[i] > ਉੱਤਮ_val) {
            ਉੱਤਮ_val = u_val.fp32[i];
            ਉੱਤਮ_idx = (int32_t) u_idx.w[i];
        }
    }

    ਲਈ (uint32_t i = nvec * 32; i < n; i++) {
        ਜੇ (src[i] > ਉੱਤਮ_val) {
            ਉੱਤਮ_val = src[i];
            ਉੱਤਮ_idx = (int32_t) (ਵਿੱਥ + i);
        }
    }

    *ਬਾਹਰਲਾ_val = ਉੱਤਮ_val;
    *ਬਾਹਰਲਾ_idx = ਉੱਤਮ_idx;
}

#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ hvx_reduce_loop_body
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HVX_REDUCE_MAX_OP
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HVX_REDUCE_ਜੋੜ_OP
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HVX_REDUCE_MAX_SCALAR
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HVX_REDUCE_ਜੋੜ_SCALAR
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HVX_ਜੋੜ_SQ_OP

#ਜੇ_ਅੰਤ /* HVX_REDUCE_H */
