#ifndef HVX_REDUCE_H
#define HVX_REDUCE_H

#অন্তর্ভাব <math.h>
#অন্তর্ভাব <stdbool.h>
#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব <assert.h>

#অন্তর্ভাব "hex-utils.h"
#অন্তর্ভাব "hvx-base.h"
#অন্তর্ভাব "hvx-types.h"

স্থির অন্তঃরেখ HVX_Vector hvx_vec_reduce_যোগ_n_i32(HVX_Vector in, unsigned পূর্ণসংখ্যা n) {
    unsigned পূর্ণসংখ্যা যোগ = n * 4;  // total vec nbytes
    unsigned পূর্ণসংখ্যা width = 4;      // int32

    HVX_Vector যোগ_2 = in, যোগ_t;
    যখন (width < যোগ) {
        যোগ_t = Q6_V_vror_VR(যোগ_2, width);     // rotate right
        যোগ_2   = Q6_Vw_vadd_VwVw(যোগ_t, যোগ_2);  // elementwise sum
        width = width << 1;
    }
    ফেরত যোগ_2;
}

স্থির অন্তঃরেখ HVX_Vector hvx_vec_reduce_যোগ_i32(HVX_Vector in) {
    ফেরত hvx_vec_reduce_যোগ_n_i32(in, 32);
}

স্থির অন্তঃরেখ HVX_Vector hvx_vec_reduce_যোগ_n_qf32(HVX_Vector in, unsigned পূর্ণসংখ্যা n) {
    unsigned পূর্ণসংখ্যা যোগ = n * 4;  // total vec nbytes
    unsigned পূর্ণসংখ্যা width = 4;      // fp32 nbytes

    HVX_Vector যোগ_2 = in, যোগ_t;
    যখন (width < যোগ) {
        যোগ_t = Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(যোগ_2), width);  // rotate right
        যোগ_2   = Q6_Vqf32_vadd_Vqf32Vsf(যোগ_2, যোগ_t);             // elementwise sum
        width = width << 1;
    }
    ফেরত যোগ_2;
}

স্থির অন্তঃরেখ HVX_Vector hvx_vec_reduce_যোগ_qf32(HVX_Vector in) {
    ফেরত hvx_vec_reduce_যোগ_n_qf32(in, 32);
}

#যদি __HVX_ARCH__ > 75

স্থির অন্তঃরেখ HVX_Vector hvx_vec_reduce_যোগ_f32x4(HVX_Vector_x4 in) {
    HVX_VectorPair যোগ_p01 = Q6_W_vshuff_VVR(in.v[1], in.v[0], 4);
    HVX_VectorPair যোগ_p23 = Q6_W_vshuff_VVR(in.v[3], in.v[2], 4);
    HVX_Vector  যোগ_sf01  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(যোগ_p01), Q6_V_hi_W(যোগ_p01));
    HVX_Vector  যোগ_sf23  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(যোগ_p23), Q6_V_hi_W(যোগ_p23));

    HVX_VectorPair যোগ_p0123 = Q6_W_vshuff_VVR(যোগ_sf23, যোগ_sf01, 8);
    HVX_Vector  যোগ_sf       = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(যোগ_p0123), Q6_V_hi_W(যোগ_p0123));

    যোগ_sf = Q6_Vsf_vadd_VsfVsf(যোগ_sf, Q6_V_vror_VR(যোগ_sf, VLEN / 2));
    যোগ_sf = Q6_Vsf_vadd_VsfVsf(যোগ_sf, Q6_V_vror_VR(যোগ_sf, VLEN / 4));
    যোগ_sf = Q6_Vsf_vadd_VsfVsf(যোগ_sf, Q6_V_vror_VR(যোগ_sf, VLEN / 8));
    ফেরত যোগ_sf;
}

স্থির অন্তঃরেখ HVX_Vector hvx_vec_reduce_যোগ_f32x2(HVX_Vector in0, HVX_Vector in1) {
    HVX_VectorPair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_Vector  যোগ_sf  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    যোগ_sf = Q6_Vsf_vadd_VsfVsf(যোগ_sf, Q6_V_vror_VR(যোগ_sf, VLEN / 2));
    যোগ_sf = Q6_Vsf_vadd_VsfVsf(যোগ_sf, Q6_V_vror_VR(যোগ_sf, VLEN / 4));
    যোগ_sf = Q6_Vsf_vadd_VsfVsf(যোগ_sf, Q6_V_vror_VR(যোগ_sf, VLEN / 8));
    যোগ_sf = Q6_Vsf_vadd_VsfVsf(যোগ_sf, Q6_V_vror_VR(যোগ_sf, VLEN / 16));
    ফেরত যোগ_sf;
}

স্থির অন্তঃরেখ HVX_Vector hvx_vec_reduce_যোগ_n_f32(HVX_Vector in, unsigned পূর্ণসংখ্যা n) {
    unsigned পূর্ণসংখ্যা যোগ = n * 4;  // total vec nbytes
    unsigned পূর্ণসংখ্যা width = 4;      // fp32 nbytes

    HVX_Vector যোগ_2 = in, যোগ_t;
    যখন (width < যোগ) {
        যোগ_t = Q6_V_vror_VR(যোগ_2, width);       // rotate right
        যোগ_2   = Q6_Vsf_vadd_VsfVsf(যোগ_2, যোগ_t); // elementwise sum
        width = width << 1;
    }
    ফেরত যোগ_2;
}

#অন্যথা

স্থির অন্তঃরেখ HVX_Vector hvx_vec_reduce_যোগ_f32x4(HVX_Vector_x4 in) {
    HVX_VectorPair যোগ_p01  = Q6_W_vshuff_VVR(in.v[1], in.v[0], 4);
    HVX_VectorPair যোগ_p23  = Q6_W_vshuff_VVR(in.v[3], in.v[2], 4);
    HVX_Vector     যোগ_qf01 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(যোগ_p01), Q6_V_hi_W(যোগ_p01));
    HVX_Vector     যোগ_qf23 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(যোগ_p23), Q6_V_hi_W(যোগ_p23));

    HVX_VectorPair যোগ_p0123 = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(যোগ_qf23), Q6_Vsf_equals_Vqf32(যোগ_qf01), 8);
    HVX_Vector     যোগ_qf    = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(যোগ_p0123), Q6_V_hi_W(যোগ_p0123));

    যোগ_qf = Q6_Vqf32_vadd_Vqf32Vsf(যোগ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(যোগ_qf), VLEN / 2));
    যোগ_qf = Q6_Vqf32_vadd_Vqf32Vsf(যোগ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(যোগ_qf), VLEN / 4));
    যোগ_qf = Q6_Vqf32_vadd_Vqf32Vsf(যোগ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(যোগ_qf), VLEN / 8));
    ফেরত Q6_Vsf_equals_Vqf32(যোগ_qf);
}

স্থির অন্তঃরেখ HVX_Vector hvx_vec_reduce_যোগ_f32x2(HVX_Vector in0, HVX_Vector in1) {
    HVX_VectorPair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_Vector  যোগ_qf  = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    যোগ_qf = Q6_Vqf32_vadd_Vqf32Vsf(যোগ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(যোগ_qf), VLEN / 2));
    যোগ_qf = Q6_Vqf32_vadd_Vqf32Vsf(যোগ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(যোগ_qf), VLEN / 4));
    যোগ_qf = Q6_Vqf32_vadd_Vqf32Vsf(যোগ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(যোগ_qf), VLEN / 8));
    যোগ_qf = Q6_Vqf32_vadd_Vqf32Vsf(যোগ_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(যোগ_qf), VLEN / 16));
    ফেরত Q6_Vsf_equals_Vqf32(যোগ_qf);
}

স্থির অন্তঃরেখ HVX_Vector hvx_vec_reduce_যোগ_n_f32(HVX_Vector in, unsigned পূর্ণসংখ্যা n) {
    unsigned পূর্ণসংখ্যা যোগ = n * 4;  // total vec nbytes
    unsigned পূর্ণসংখ্যা width = 4;      // fp32 nbytes

    HVX_Vector যোগ_2 = in, যোগ_t;
    যখন (width < যোগ) {
        যোগ_t = Q6_V_vror_VR(যোগ_2, width);                               // rotate right
        যোগ_2   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(যোগ_2, যোগ_t));  // elementwise sum
        width = width << 1;
    }
    ফেরত যোগ_2;
}

#endif

স্থির অন্তঃরেখ HVX_Vector hvx_vec_reduce_যোগ_f32(HVX_Vector in) {
    ফেরত hvx_vec_reduce_যোগ_n_f32(in, 32);
}

স্থির অন্তঃরেখ HVX_Vector hvx_vec_reduce_max_f16(HVX_Vector in) {
    unsigned যোগ = 128;  // total vec nbytes
    unsigned width = 2;    // fp16 nbytes

    HVX_Vector _max = in, _max_t;
    যখন (width < যোগ) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    ফেরত _max;
}

স্থির অন্তঃরেখ HVX_Vector hvx_vec_reduce_max2_f16(HVX_Vector in, HVX_Vector _max) {
    unsigned যোগ = 128;  // total vec nbytes
    unsigned width = 2;    // fp32 nbytes

    HVX_Vector _max_t;

    _max = Q6_Vhf_vmax_VhfVhf(in, _max);
    যখন (width < যোগ) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    ফেরত _max;
}

স্থির অন্তঃরেখ HVX_Vector hvx_vec_reduce_max_f32(HVX_Vector in) {
    unsigned যোগ = 128;  // total vec nbytes
    unsigned width = 4;    // fp32 nbytes

    HVX_Vector _max = in, _max_t;
    যখন (width < যোগ) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    ফেরত _max;
}

স্থির অন্তঃরেখ HVX_Vector hvx_vec_reduce_max2_f32(HVX_Vector in, HVX_Vector _max) {
    unsigned যোগ = 128;  // total vec nbytes
    unsigned width = 4;    // fp32 nbytes

    HVX_Vector _max_t;

    _max = Q6_Vsf_vmax_VsfVsf(in, _max);
    যখন (width < যোগ) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    ফেরত _max;
}

#define hvx_reduce_loop_body(src_type, init_vec, pad_vec, vec_op, reduce_op, scalar_reduce) \
    করো {                                                                                    \
        src_type * restrict vsrc = (src_type *) src;                                        \
        HVX_Vector acc = init_vec;                                                          \
                                                                                            \
        ধ্রুবক uint32_t elem_আকার = আকার(দশমিক);                                           \
        ধ্রুবক uint32_t epv  = 128 / elem_আকার;                                              \
        ধ্রুবক uint32_t nvec = অঙ্ক_elems / epv;                                              \
        ধ্রুবক uint32_t nloe = অঙ্ক_elems % epv;                                              \
                                                                                            \
        uint32_t i = 0;                                                                     \
        _Pragma("unroll(4)")                                                                \
        জন্য (; i < nvec; i++) {                                                             \
            acc = vec_op(acc, vsrc[i]);                                                     \
        }                                                                                   \
        যদি (nloe) {                                                                         \
            ধ্রুবক দশমিক * srcf = (ধ্রুবক দশমিক *) src + i * epv;                             \
            HVX_Vector in = *(HVX_UVector *) srcf;                                          \
            HVX_Vector অস্থায়ী = Q6_V_valign_VVR(in, pad_vec, nloe * elem_আকার);               \
            acc = vec_op(acc, অস্থায়ী);                                                        \
        }                                                                                   \
        HVX_Vector v = reduce_op(acc);                                                      \
        ফেরত scalar_reduce(v);                                                            \
    } যখন(0)

#define HVX_REDUCE_MAX_OP(acc, val) Q6_Vsf_vmax_VsfVsf(acc, val)
#define HVX_REDUCE_যোগ_OP(acc, val) Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(acc), val)
#define HVX_যোগ_SQ_OP(acc, val) Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(val, val))
#define HVX_REDUCE_MAX_SCALAR(v) hvx_vec_get_f32(v)
#define HVX_REDUCE_যোগ_SCALAR(v) hvx_vec_get_f32(Q6_Vsf_equals_Vqf32(v))

// Max variants

স্থির অন্তঃরেখ দশমিক hvx_reduce_max_f32_a(ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems) {
    HVX_Vector init_vec = hvx_vec_splat_f32(((ধ্রুবক দশমিক *) src)[0]);
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_reduce_loop_body(HVX_Vector, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

স্থির অন্তঃরেখ দশমিক hvx_reduce_max_f32_u(ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems) {
    HVX_Vector init_vec = hvx_vec_splat_f32(((ধ্রুবক দশমিক *) src)[0]);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

স্থির অন্তঃরেখ দশমিক hvx_reduce_max_f32(ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems) {
    যদি (hex_is_aligned((শূন্য *) src, 128)) {
        ফেরত hvx_reduce_max_f32_a(src, অঙ্ক_elems);
    } অন্যথা {
        ফেরত hvx_reduce_max_f32_u(src, অঙ্ক_elems);
    }
}

// Sum variants

স্থির অন্তঃরেখ দশমিক hvx_reduce_যোগ_f32_a(ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(0);
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_reduce_loop_body(HVX_Vector, init_vec, init_vec, HVX_REDUCE_যোগ_OP, hvx_vec_reduce_যোগ_qf32, HVX_REDUCE_যোগ_SCALAR);
}

স্থির অন্তঃরেখ দশমিক hvx_reduce_যোগ_f32_u(ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_যোগ_OP, hvx_vec_reduce_যোগ_qf32, HVX_REDUCE_যোগ_SCALAR);
}

স্থির অন্তঃরেখ দশমিক hvx_reduce_যোগ_f32(ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems) {
    যদি (hex_is_aligned((শূন্য *) src, 128)) {
        ফেরত hvx_reduce_যোগ_f32_a(src, অঙ্ক_elems);
    } অন্যথা {
        ফেরত hvx_reduce_যোগ_f32_u(src, অঙ্ক_elems);
    }
}

// Sum of squares variants

স্থির অন্তঃরেখ দশমিক hvx_যোগ_of_squares_f32_a(ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(0);
    প্রতিজ্ঞা((uintptr_t) src % 128 == 0);
    hvx_reduce_loop_body(HVX_Vector, init_vec, init_vec, HVX_যোগ_SQ_OP, hvx_vec_reduce_যোগ_qf32, HVX_REDUCE_যোগ_SCALAR);
}

স্থির অন্তঃরেখ দশমিক hvx_যোগ_of_squares_f32_u(ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_যোগ_SQ_OP, hvx_vec_reduce_যোগ_qf32, HVX_REDUCE_যোগ_SCALAR);
}

স্থির অন্তঃরেখ দশমিক hvx_যোগ_of_squares_f32(ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems) {
    যদি (hex_is_aligned((শূন্য *) src, 128)) {
        ফেরত hvx_যোগ_of_squares_f32_a(src, অঙ্ক_elems);
    } অন্যথা {
        ফেরত hvx_যোগ_of_squares_f32_u(src, অঙ্ক_elems);
    }
}

// Signed 32-bit Integer Max variants

স্থির অন্তঃরেখ HVX_Vector hvx_vec_reduce_max_n_i32(HVX_Vector in, unsigned পূর্ণসংখ্যা n) {
    unsigned পূর্ণসংখ্যা যোগ = n * 4;  // total vec nbytes
    unsigned পূর্ণসংখ্যা width = 4;      // int32 nbytes

    HVX_Vector max_val = in, max_t;
    যখন (width < যোগ) {
        max_t = Q6_V_vror_VR(max_val, width);          // rotate right
        max_val = Q6_Vw_vmax_VwVw(max_t, max_val);      // elementwise signed max
        width = width << 1;
    }
    ফেরত max_val;
}

স্থির অন্তঃরেখ HVX_Vector hvx_vec_reduce_max_i32(HVX_Vector in) {
    ফেরত hvx_vec_reduce_max_n_i32(in, 32);
}

স্থির অন্তঃরেখ int32_t hvx_reduce_max_i32_a(ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(((ধ্রুবক int32_t *) src)[0]);
    HVX_Vector pad_vec  = Q6_V_vsplat_R(0x80000000);
    প্রতিজ্ঞা((uintptr_t) src % 128 == 0);
    hvx_reduce_loop_body(HVX_Vector, init_vec, pad_vec, Q6_Vw_vmax_VwVw, hvx_vec_reduce_max_i32, hvx_vec_get_i32);
}

স্থির অন্তঃরেখ int32_t hvx_reduce_max_i32_u(ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(((ধ্রুবক int32_t *) src)[0]);
    HVX_Vector pad_vec  = Q6_V_vsplat_R(0x80000000);
    hvx_reduce_loop_body(HVX_UVector, init_vec, pad_vec, Q6_Vw_vmax_VwVw, hvx_vec_reduce_max_i32, hvx_vec_get_i32);
}

স্থির অন্তঃরেখ int32_t hvx_reduce_max_i32(ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems) {
    যদি (hex_is_aligned((শূন্য *) src, 128)) {
        ফেরত hvx_reduce_max_i32_a(src, অঙ্ক_elems);
    } অন্যথা {
        ফেরত hvx_reduce_max_i32_u(src, অঙ্ক_elems);
    }
}

স্থির অন্তঃরেখ শূন্য hvx_argmax_f32(
    ধ্রুবক দশমিক * restrict src,
    uint32_t n,
    uint32_t offset,
    দশমিক * out_val,
    int32_t * out_idx
) {
    যদি (n == 0) {
        *out_val = -INFINITY;
        *out_idx = (int32_t) offset;
        ফেরত;
    }

    যদি (n < 32 || !hex_is_aligned((শূন্য *) src, 128)) {
        দশমিক best_val = src[0];
        int32_t best_idx = (int32_t) offset;
        জন্য (uint32_t i = 1; i < n; i++) {
            যদি (src[i] > best_val) {
                best_val = src[i];
                best_idx = (int32_t) (offset + i);
            }
        }
        *out_val = best_val;
        *out_idx = best_idx;
        ফেরত;
    }

    স্থির ধ্রুবক int32_t c_lane_idx[32] __attribute__((aligned(128))) = {
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    };

    ধ্রুবক HVX_Vector v_init_idx = *(ধ্রুবক HVX_Vector *) c_lane_idx;
    ধ্রুবক HVX_Vector v_step     = Q6_V_vsplat_R(32);
    HVX_Vector v_cur_idx        = Q6_Vw_vadd_VwVw(v_init_idx, Q6_V_vsplat_R((int32_t) offset));
    HVX_Vector v_max_val        = hvx_vec_splat_f32(-INFINITY);
    HVX_Vector v_max_idx        = v_cur_idx;

    ধ্রুবক HVX_Vector * vsrc = (ধ্রুবক HVX_Vector *) src;
    ধ্রুবক uint32_t nvec = n / 32;

    জন্য (uint32_t vi = 0; vi < nvec; vi++) {
        HVX_Vector v = vsrc[vi];
        HVX_VectorPred pred = Q6_Q_vcmp_gt_VsfVsf(v, v_max_val);
        v_max_val = Q6_V_vmux_QVV(pred, v, v_max_val);
        v_max_idx = Q6_V_vmux_QVV(pred, v_cur_idx, v_max_idx);
        v_cur_idx = Q6_Vw_vadd_VwVw(v_cur_idx, v_step);
    }

    HVX_VectorAlias u_val, u_idx;
    u_val.v = v_max_val;
    u_idx.v = v_max_idx;

    দশমিক best_val = u_val.fp32[0];
    int32_t best_idx = (int32_t) u_idx.w[0];
    জন্য (পূর্ণসংখ্যা i = 1; i < 32; i++) {
        যদি (u_val.fp32[i] > best_val) {
            best_val = u_val.fp32[i];
            best_idx = (int32_t) u_idx.w[i];
        }
    }

    জন্য (uint32_t i = nvec * 32; i < n; i++) {
        যদি (src[i] > best_val) {
            best_val = src[i];
            best_idx = (int32_t) (offset + i);
        }
    }

    *out_val = best_val;
    *out_idx = best_idx;
}

#undef hvx_reduce_loop_body
#undef HVX_REDUCE_MAX_OP
#undef HVX_REDUCE_যোগ_OP
#undef HVX_REDUCE_MAX_SCALAR
#undef HVX_REDUCE_যোগ_SCALAR
#undef HVX_যোগ_SQ_OP

#endif /* HVX_REDUCE_H */
