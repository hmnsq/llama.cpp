#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HVX_REDUCE_H
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_REDUCE_H

#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <assert.h>

#ಅನ್ತರ್ಭಾವ "hex-utils.h"
#ಅನ್ತರ್ಭಾವ "hvx-base.h"
#ಅನ್ತರ್ಭಾವ "hvx-types.h"

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_reduce_ಮೊತ್ತ_n_i32(HVX_ಸದಿಶ ಒಳ, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n) {
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ಒಟ್ಟು = n * 4;  // total vec nbytes
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ಅಗಲ = 4;      // int32

    HVX_ಸದಿಶ ಮೊತ್ತ = ಒಳ, ಮೊತ್ತ_t;
    ಆಗ (ಅಗಲ < ಒಟ್ಟು) {
        ಮೊತ್ತ_t = Q6_V_vror_VR(ಮೊತ್ತ, ಅಗಲ);     // rotate right
        ಮೊತ್ತ   = Q6_Vw_vadd_VwVw(ಮೊತ್ತ_t, ಮೊತ್ತ);  // elementwise sum
        ಅಗಲ = ಅಗಲ << 1;
    }
    ಹಿಂತಿರುಗಿ ಮೊತ್ತ;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_reduce_ಮೊತ್ತ_i32(HVX_ಸದಿಶ ಒಳ) {
    ಹಿಂತಿರುಗಿ hvx_vec_reduce_ಮೊತ್ತ_n_i32(ಒಳ, 32);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_reduce_ಮೊತ್ತ_n_qf32(HVX_ಸದಿಶ ಒಳ, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n) {
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ಒಟ್ಟು = n * 4;  // total vec nbytes
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ಅಗಲ = 4;      // fp32 nbytes

    HVX_ಸದಿಶ ಮೊತ್ತ = ಒಳ, ಮೊತ್ತ_t;
    ಆಗ (ಅಗಲ < ಒಟ್ಟು) {
        ಮೊತ್ತ_t = Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(ಮೊತ್ತ), ಅಗಲ);  // rotate right
        ಮೊತ್ತ   = Q6_Vqf32_vadd_Vqf32Vsf(ಮೊತ್ತ, ಮೊತ್ತ_t);             // elementwise sum
        ಅಗಲ = ಅಗಲ << 1;
    }
    ಹಿಂತಿರುಗಿ ಮೊತ್ತ;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_reduce_ಮೊತ್ತ_qf32(HVX_ಸದಿಶ ಒಳ) {
    ಹಿಂತಿರುಗಿ hvx_vec_reduce_ಮೊತ್ತ_n_qf32(ಒಳ, 32);
}

#ಆದರೆ __HVX_ARCH__ > 75

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_reduce_ಮೊತ್ತ_f32x4(HVX_ಸದಿಶ_x4 ಒಳ) {
    HVX_ಸದಿಶ_Pair ಮೊತ್ತ_p01 = Q6_W_vshuff_VVR(ಒಳ.v[1], ಒಳ.v[0], 4);
    HVX_ಸದಿಶ_Pair ಮೊತ್ತ_p23 = Q6_W_vshuff_VVR(ಒಳ.v[3], ಒಳ.v[2], 4);
    HVX_ಸದಿಶ  ಮೊತ್ತ_sf01  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(ಮೊತ್ತ_p01), Q6_V_hi_W(ಮೊತ್ತ_p01));
    HVX_ಸದಿಶ  ಮೊತ್ತ_sf23  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(ಮೊತ್ತ_p23), Q6_V_hi_W(ಮೊತ್ತ_p23));

    HVX_ಸದಿಶ_Pair ಮೊತ್ತ_p0123 = Q6_W_vshuff_VVR(ಮೊತ್ತ_sf23, ಮೊತ್ತ_sf01, 8);
    HVX_ಸದಿಶ  ಮೊತ್ತ_sf       = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(ಮೊತ್ತ_p0123), Q6_V_hi_W(ಮೊತ್ತ_p0123));

    ಮೊತ್ತ_sf = Q6_Vsf_vadd_VsfVsf(ಮೊತ್ತ_sf, Q6_V_vror_VR(ಮೊತ್ತ_sf, VLEN / 2));
    ಮೊತ್ತ_sf = Q6_Vsf_vadd_VsfVsf(ಮೊತ್ತ_sf, Q6_V_vror_VR(ಮೊತ್ತ_sf, VLEN / 4));
    ಮೊತ್ತ_sf = Q6_Vsf_vadd_VsfVsf(ಮೊತ್ತ_sf, Q6_V_vror_VR(ಮೊತ್ತ_sf, VLEN / 8));
    ಹಿಂತಿರುಗಿ ಮೊತ್ತ_sf;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_reduce_ಮೊತ್ತ_f32x2(HVX_ಸದಿಶ in0, HVX_ಸದಿಶ in1) {
    HVX_ಸದಿಶ_Pair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_ಸದಿಶ  ಮೊತ್ತ_sf  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    ಮೊತ್ತ_sf = Q6_Vsf_vadd_VsfVsf(ಮೊತ್ತ_sf, Q6_V_vror_VR(ಮೊತ್ತ_sf, VLEN / 2));
    ಮೊತ್ತ_sf = Q6_Vsf_vadd_VsfVsf(ಮೊತ್ತ_sf, Q6_V_vror_VR(ಮೊತ್ತ_sf, VLEN / 4));
    ಮೊತ್ತ_sf = Q6_Vsf_vadd_VsfVsf(ಮೊತ್ತ_sf, Q6_V_vror_VR(ಮೊತ್ತ_sf, VLEN / 8));
    ಮೊತ್ತ_sf = Q6_Vsf_vadd_VsfVsf(ಮೊತ್ತ_sf, Q6_V_vror_VR(ಮೊತ್ತ_sf, VLEN / 16));
    ಹಿಂತಿರುಗಿ ಮೊತ್ತ_sf;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_reduce_ಮೊತ್ತ_n_f32(HVX_ಸದಿಶ ಒಳ, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n) {
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ಒಟ್ಟು = n * 4;  // total vec nbytes
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ಅಗಲ = 4;      // fp32 nbytes

    HVX_ಸದಿಶ ಮೊತ್ತ = ಒಳ, ಮೊತ್ತ_t;
    ಆಗ (ಅಗಲ < ಒಟ್ಟು) {
        ಮೊತ್ತ_t = Q6_V_vror_VR(ಮೊತ್ತ, ಅಗಲ);       // rotate right
        ಮೊತ್ತ   = Q6_Vsf_vadd_VsfVsf(ಮೊತ್ತ, ಮೊತ್ತ_t); // elementwise sum
        ಅಗಲ = ಅಗಲ << 1;
    }
    ಹಿಂತಿರುಗಿ ಮೊತ್ತ;
}

#ಇಲ್ಲದಿದ್ದರೆ

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_reduce_ಮೊತ್ತ_f32x4(HVX_ಸದಿಶ_x4 ಒಳ) {
    HVX_ಸದಿಶ_Pair ಮೊತ್ತ_p01  = Q6_W_vshuff_VVR(ಒಳ.v[1], ಒಳ.v[0], 4);
    HVX_ಸದಿಶ_Pair ಮೊತ್ತ_p23  = Q6_W_vshuff_VVR(ಒಳ.v[3], ಒಳ.v[2], 4);
    HVX_ಸದಿಶ     ಮೊತ್ತ_qf01 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(ಮೊತ್ತ_p01), Q6_V_hi_W(ಮೊತ್ತ_p01));
    HVX_ಸದಿಶ     ಮೊತ್ತ_qf23 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(ಮೊತ್ತ_p23), Q6_V_hi_W(ಮೊತ್ತ_p23));

    HVX_ಸದಿಶ_Pair ಮೊತ್ತ_p0123 = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(ಮೊತ್ತ_qf23), Q6_Vsf_equals_Vqf32(ಮೊತ್ತ_qf01), 8);
    HVX_ಸದಿಶ     ಮೊತ್ತ_qf    = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(ಮೊತ್ತ_p0123), Q6_V_hi_W(ಮೊತ್ತ_p0123));

    ಮೊತ್ತ_qf = Q6_Vqf32_vadd_Vqf32Vsf(ಮೊತ್ತ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(ಮೊತ್ತ_qf), VLEN / 2));
    ಮೊತ್ತ_qf = Q6_Vqf32_vadd_Vqf32Vsf(ಮೊತ್ತ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(ಮೊತ್ತ_qf), VLEN / 4));
    ಮೊತ್ತ_qf = Q6_Vqf32_vadd_Vqf32Vsf(ಮೊತ್ತ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(ಮೊತ್ತ_qf), VLEN / 8));
    ಹಿಂತಿರುಗಿ Q6_Vsf_equals_Vqf32(ಮೊತ್ತ_qf);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_reduce_ಮೊತ್ತ_f32x2(HVX_ಸದಿಶ in0, HVX_ಸದಿಶ in1) {
    HVX_ಸದಿಶ_Pair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_ಸದಿಶ  ಮೊತ್ತ_qf  = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    ಮೊತ್ತ_qf = Q6_Vqf32_vadd_Vqf32Vsf(ಮೊತ್ತ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(ಮೊತ್ತ_qf), VLEN / 2));
    ಮೊತ್ತ_qf = Q6_Vqf32_vadd_Vqf32Vsf(ಮೊತ್ತ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(ಮೊತ್ತ_qf), VLEN / 4));
    ಮೊತ್ತ_qf = Q6_Vqf32_vadd_Vqf32Vsf(ಮೊತ್ತ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(ಮೊತ್ತ_qf), VLEN / 8));
    ಮೊತ್ತ_qf = Q6_Vqf32_vadd_Vqf32Vsf(ಮೊತ್ತ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(ಮೊತ್ತ_qf), VLEN / 16));
    ಹಿಂತಿರುಗಿ Q6_Vsf_equals_Vqf32(ಮೊತ್ತ_qf);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_reduce_ಮೊತ್ತ_n_f32(HVX_ಸದಿಶ ಒಳ, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n) {
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ಒಟ್ಟು = n * 4;  // total vec nbytes
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ಅಗಲ = 4;      // fp32 nbytes

    HVX_ಸದಿಶ ಮೊತ್ತ = ಒಳ, ಮೊತ್ತ_t;
    ಆಗ (ಅಗಲ < ಒಟ್ಟು) {
        ಮೊತ್ತ_t = Q6_V_vror_VR(ಮೊತ್ತ, ಅಗಲ);                               // rotate right
        ಮೊತ್ತ   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(ಮೊತ್ತ, ಮೊತ್ತ_t));  // elementwise sum
        ಅಗಲ = ಅಗಲ << 1;
    }
    ಹಿಂತಿರುಗಿ ಮೊತ್ತ;
}

#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_reduce_ಮೊತ್ತ_f32(HVX_ಸದಿಶ ಒಳ) {
    ಹಿಂತಿರುಗಿ hvx_vec_reduce_ಮೊತ್ತ_n_f32(ಒಳ, 32);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_reduce_max_f16(HVX_ಸದಿಶ ಒಳ) {
    ಚಿಹ್ನೆರಹಿತ ಒಟ್ಟು = 128;  // total vec nbytes
    ಚಿಹ್ನೆರಹಿತ ಅಗಲ = 2;    // fp16 nbytes

    HVX_ಸದಿಶ _max = ಒಳ, _max_t;
    ಆಗ (ಅಗಲ < ಒಟ್ಟು) {
        _max_t = Q6_V_vror_VR(_max, ಅಗಲ);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        ಅಗಲ  = ಅಗಲ << 1;
    }

    ಹಿಂತಿರುಗಿ _max;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_reduce_max2_f16(HVX_ಸದಿಶ ಒಳ, HVX_ಸದಿಶ _max) {
    ಚಿಹ್ನೆರಹಿತ ಒಟ್ಟು = 128;  // total vec nbytes
    ಚಿಹ್ನೆರಹಿತ ಅಗಲ = 2;    // fp32 nbytes

    HVX_ಸದಿಶ _max_t;

    _max = Q6_Vhf_vmax_VhfVhf(ಒಳ, _max);
    ಆಗ (ಅಗಲ < ಒಟ್ಟು) {
        _max_t = Q6_V_vror_VR(_max, ಅಗಲ);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        ಅಗಲ  = ಅಗಲ << 1;
    }

    ಹಿಂತಿರುಗಿ _max;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_reduce_max_f32(HVX_ಸದಿಶ ಒಳ) {
    ಚಿಹ್ನೆರಹಿತ ಒಟ್ಟು = 128;  // total vec nbytes
    ಚಿಹ್ನೆರಹಿತ ಅಗಲ = 4;    // fp32 nbytes

    HVX_ಸದಿಶ _max = ಒಳ, _max_t;
    ಆಗ (ಅಗಲ < ಒಟ್ಟು) {
        _max_t = Q6_V_vror_VR(_max, ಅಗಲ);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        ಅಗಲ  = ಅಗಲ << 1;
    }

    ಹಿಂತಿರುಗಿ _max;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_reduce_max2_f32(HVX_ಸದಿಶ ಒಳ, HVX_ಸದಿಶ _max) {
    ಚಿಹ್ನೆರಹಿತ ಒಟ್ಟು = 128;  // total vec nbytes
    ಚಿಹ್ನೆರಹಿತ ಅಗಲ = 4;    // fp32 nbytes

    HVX_ಸದಿಶ _max_t;

    _max = Q6_Vsf_vmax_VsfVsf(ಒಳ, _max);
    ಆಗ (ಅಗಲ < ಒಟ್ಟು) {
        _max_t = Q6_V_vror_VR(_max, ಅಗಲ);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        ಅಗಲ  = ಅಗಲ << 1;
    }

    ಹಿಂತಿರುಗಿ _max;
}

#ವ್ಯಾಖ್ಯಾನಿಸು hvx_reduce_loop_body(src_type, init_vec, pad_vec, vec_op, reduce_op, scalar_reduce) \
    ಮಾಡು {                                                                                    \
        src_type * ಸೀಮಿತ vsrc = (src_type *) src;                                        \
        HVX_ಸದಿಶ acc = init_vec;                                                          \
                                                                                            \
        ಸ್ಥಿರಾಂಕ uint32_t elem_ಗಾತ್ರ = ಗಾತ್ರ(ದಶಮಾಂಶ);                                           \
        ಸ್ಥಿರಾಂಕ uint32_t epv  = 128 / elem_ಗಾತ್ರ;                                              \
        ಸ್ಥಿರಾಂಕ uint32_t nvec = ಅಂಕೆ_elems / epv;                                              \
        ಸ್ಥಿರಾಂಕ uint32_t nloe = ಅಂಕೆ_elems % epv;                                              \
                                                                                            \
        uint32_t i = 0;                                                                     \
        _Pragma("unroll(4)")                                                                \
        ಗಾಗಿ (; i < nvec; i++) {                                                             \
            acc = vec_op(acc, vsrc[i]);                                                     \
        }                                                                                   \
        ಆದರೆ (nloe) {                                                                         \
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * srcf = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src + i * epv;                             \
            HVX_ಸದಿಶ ಒಳ = *(HVX_UVector *) srcf;                                          \
            HVX_ಸದಿಶ ತಾತ್ಕಾಲಿಕ = Q6_V_valign_VVR(ಒಳ, pad_vec, nloe * elem_ಗಾತ್ರ);               \
            acc = vec_op(acc, ತಾತ್ಕಾಲಿಕ);                                                        \
        }                                                                                   \
        HVX_ಸದಿಶ v = reduce_op(acc);                                                      \
        ಹಿಂತಿರುಗಿ scalar_reduce(v);                                                            \
    } ಆಗ(0)

#ವ್ಯಾಖ್ಯಾನಿಸು HVX_REDUCE_MAX_OP(acc, val) Q6_Vsf_vmax_VsfVsf(acc, val)
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_REDUCE_ಮೊತ್ತ_OP(acc, val) Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(acc), val)
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_ಮೊತ್ತ_SQ_OP(acc, val) Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(val, val))
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_REDUCE_MAX_SCALAR(v) hvx_vec_get_f32(v)
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_REDUCE_ಮೊತ್ತ_SCALAR(v) hvx_vec_get_f32(Q6_Vsf_equals_Vqf32(v))

// Max variants

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ hvx_reduce_max_f32_a(ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems) {
    HVX_ಸದಿಶ init_vec = hvx_vec_splat_f32(((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src)[0]);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_reduce_loop_body(HVX_ಸದಿಶ, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ hvx_reduce_max_f32_u(ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems) {
    HVX_ಸದಿಶ init_vec = hvx_vec_splat_f32(((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src)[0]);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ hvx_reduce_max_f32(ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems) {
    ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        ಹಿಂತಿರುಗಿ hvx_reduce_max_f32_a(src, ಅಂಕೆ_elems);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ hvx_reduce_max_f32_u(src, ಅಂಕೆ_elems);
    }
}

// Sum variants

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ hvx_reduce_ಮೊತ್ತ_f32_a(ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems) {
    HVX_ಸದಿಶ init_vec = Q6_V_vsplat_R(0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_reduce_loop_body(HVX_ಸದಿಶ, init_vec, init_vec, HVX_REDUCE_ಮೊತ್ತ_OP, hvx_vec_reduce_ಮೊತ್ತ_qf32, HVX_REDUCE_ಮೊತ್ತ_SCALAR);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ hvx_reduce_ಮೊತ್ತ_f32_u(ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems) {
    HVX_ಸದಿಶ init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_ಮೊತ್ತ_OP, hvx_vec_reduce_ಮೊತ್ತ_qf32, HVX_REDUCE_ಮೊತ್ತ_SCALAR);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ hvx_reduce_ಮೊತ್ತ_f32(ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems) {
    ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        ಹಿಂತಿರುಗಿ hvx_reduce_ಮೊತ್ತ_f32_a(src, ಅಂಕೆ_elems);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ hvx_reduce_ಮೊತ್ತ_f32_u(src, ಅಂಕೆ_elems);
    }
}

// Sum of squares variants

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ hvx_ಮೊತ್ತ_ದ_squares_f32_a(ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems) {
    HVX_ಸದಿಶ init_vec = Q6_V_vsplat_R(0);
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src % 128 == 0);
    hvx_reduce_loop_body(HVX_ಸದಿಶ, init_vec, init_vec, HVX_ಮೊತ್ತ_SQ_OP, hvx_vec_reduce_ಮೊತ್ತ_qf32, HVX_REDUCE_ಮೊತ್ತ_SCALAR);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ hvx_ಮೊತ್ತ_ದ_squares_f32_u(ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems) {
    HVX_ಸದಿಶ init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_ಮೊತ್ತ_SQ_OP, hvx_vec_reduce_ಮೊತ್ತ_qf32, HVX_REDUCE_ಮೊತ್ತ_SCALAR);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ hvx_ಮೊತ್ತ_ದ_squares_f32(ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems) {
    ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        ಹಿಂತಿರುಗಿ hvx_ಮೊತ್ತ_ದ_squares_f32_a(src, ಅಂಕೆ_elems);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ hvx_ಮೊತ್ತ_ದ_squares_f32_u(src, ಅಂಕೆ_elems);
    }
}

// Signed 32-bit Integer Max variants

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_reduce_max_n_i32(HVX_ಸದಿಶ ಒಳ, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n) {
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ಒಟ್ಟು = n * 4;  // total vec nbytes
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ಅಗಲ = 4;      // int32 nbytes

    HVX_ಸದಿಶ max_val = ಒಳ, max_t;
    ಆಗ (ಅಗಲ < ಒಟ್ಟು) {
        max_t = Q6_V_vror_VR(max_val, ಅಗಲ);          // rotate right
        max_val = Q6_Vw_vmax_VwVw(max_t, max_val);      // elementwise signed max
        ಅಗಲ = ಅಗಲ << 1;
    }
    ಹಿಂತಿರುಗಿ max_val;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_reduce_max_i32(HVX_ಸದಿಶ ಒಳ) {
    ಹಿಂತಿರುಗಿ hvx_vec_reduce_max_n_i32(ಒಳ, 32);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ int32_t hvx_reduce_max_i32_a(ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems) {
    HVX_ಸದಿಶ init_vec = Q6_V_vsplat_R(((ಸ್ಥಿರಾಂಕ int32_t *) src)[0]);
    HVX_ಸದಿಶ pad_vec  = Q6_V_vsplat_R(0x80000000);
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src % 128 == 0);
    hvx_reduce_loop_body(HVX_ಸದಿಶ, init_vec, pad_vec, Q6_Vw_vmax_VwVw, hvx_vec_reduce_max_i32, hvx_vec_get_i32);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ int32_t hvx_reduce_max_i32_u(ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems) {
    HVX_ಸದಿಶ init_vec = Q6_V_vsplat_R(((ಸ್ಥಿರಾಂಕ int32_t *) src)[0]);
    HVX_ಸದಿಶ pad_vec  = Q6_V_vsplat_R(0x80000000);
    hvx_reduce_loop_body(HVX_UVector, init_vec, pad_vec, Q6_Vw_vmax_VwVw, hvx_vec_reduce_max_i32, hvx_vec_get_i32);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ int32_t hvx_reduce_max_i32(ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems) {
    ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        ಹಿಂತಿರುಗಿ hvx_reduce_max_i32_a(src, ಅಂಕೆ_elems);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ hvx_reduce_max_i32_u(src, ಅಂಕೆ_elems);
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_argmax_f32(
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ src,
    uint32_t n,
    uint32_t ಸ್ಥಾನಾಂತರ,
    ದಶಮಾಂಶ * ಹೊರಹೋಗು_val,
    int32_t * ಹೊರಹೋಗು_idx
) {
    ಆದರೆ (n == 0) {
        *ಹೊರಹೋಗು_val = -INFINITY;
        *ಹೊರಹೋಗು_idx = (int32_t) ಸ್ಥಾನಾಂತರ;
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (n < 32 || !hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        ದಶಮಾಂಶ ಉತ್ತಮ_val = src[0];
        int32_t ಉತ್ತಮ_idx = (int32_t) ಸ್ಥಾನಾಂತರ;
        ಗಾಗಿ (uint32_t i = 1; i < n; i++) {
            ಆದರೆ (src[i] > ಉತ್ತಮ_val) {
                ಉತ್ತಮ_val = src[i];
                ಉತ್ತಮ_idx = (int32_t) (ಸ್ಥಾನಾಂತರ + i);
            }
        }
        *ಹೊರಹೋಗು_val = ಉತ್ತಮ_val;
        *ಹೊರಹೋಗು_idx = ಉತ್ತಮ_idx;
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ int32_t c_lane_idx[32] __attribute__((aligned(128))) = {
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    };

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_init_idx = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) c_lane_idx;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_ಹೆಜ್ಜೆ     = Q6_V_vsplat_R(32);
    HVX_ಸದಿಶ v_cur_idx        = Q6_Vw_vadd_VwVw(v_init_idx, Q6_V_vsplat_R((int32_t) ಸ್ಥಾನಾಂತರ));
    HVX_ಸದಿಶ v_max_val        = hvx_vec_splat_f32(-INFINITY);
    HVX_ಸದಿಶ v_max_idx        = v_cur_idx;

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * vsrc = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) src;
    ಸ್ಥಿರಾಂಕ uint32_t nvec = n / 32;

    ಗಾಗಿ (uint32_t vi = 0; vi < nvec; vi++) {
        HVX_ಸದಿಶ v = vsrc[vi];
        HVX_ಸದಿಶ_Pred pred = Q6_Q_vcmp_gt_VsfVsf(v, v_max_val);
        v_max_val = Q6_V_vmux_QVV(pred, v, v_max_val);
        v_max_idx = Q6_V_vmux_QVV(pred, v_cur_idx, v_max_idx);
        v_cur_idx = Q6_Vw_vadd_VwVw(v_cur_idx, v_ಹೆಜ್ಜೆ);
    }

    HVX_ಸದಿಶ_Alias u_val, u_idx;
    u_val.v = v_max_val;
    u_idx.v = v_max_idx;

    ದಶಮಾಂಶ ಉತ್ತಮ_val = u_val.fp32[0];
    int32_t ಉತ್ತಮ_idx = (int32_t) u_idx.w[0];
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < 32; i++) {
        ಆದರೆ (u_val.fp32[i] > ಉತ್ತಮ_val) {
            ಉತ್ತಮ_val = u_val.fp32[i];
            ಉತ್ತಮ_idx = (int32_t) u_idx.w[i];
        }
    }

    ಗಾಗಿ (uint32_t i = nvec * 32; i < n; i++) {
        ಆದರೆ (src[i] > ಉತ್ತಮ_val) {
            ಉತ್ತಮ_val = src[i];
            ಉತ್ತಮ_idx = (int32_t) (ಸ್ಥಾನಾಂತರ + i);
        }
    }

    *ಹೊರಹೋಗು_val = ಉತ್ತಮ_val;
    *ಹೊರಹೋಗು_idx = ಉತ್ತಮ_idx;
}

#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು hvx_reduce_loop_body
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HVX_REDUCE_MAX_OP
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HVX_REDUCE_ಮೊತ್ತ_OP
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HVX_REDUCE_MAX_SCALAR
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HVX_REDUCE_ಮೊತ್ತ_SCALAR
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HVX_ಮೊತ್ತ_SQ_OP

#ಆದರೆ_ಅಂತ್ಯ /* HVX_REDUCE_H */
