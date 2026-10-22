#నిర్వచితం_కాకపోతే HVX_REDUCE_H
#నిర్వచించు HVX_REDUCE_H

#అన్తర్భావ <math.h>
#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <stdint.h>
#అన్తర్భావ <assert.h>

#అన్తర్భావ "hex-utils.h"
#అన్తర్భావ "hvx-base.h"
#అన్తర్భావ "hvx-types.h"

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_reduce_కూడిక_n_i32(HVX_సదిశ లోపల, గుర్తులేని పూర్ణాంకం n) {
    గుర్తులేని పూర్ణాంకం మొత్తం = n * 4;  // total vec nbytes
    గుర్తులేని పూర్ణాంకం వెడల్పు = 4;      // int32

    HVX_సదిశ కూడిక = లోపల, కూడిక_t;
    అయితే (వెడల్పు < మొత్తం) {
        కూడిక_t = Q6_V_vror_VR(కూడిక, వెడల్పు);     // rotate right
        కూడిక   = Q6_Vw_vadd_VwVw(కూడిక_t, కూడిక);  // elementwise sum
        వెడల్పు = వెడల్పు << 1;
    }
    తిరిగి కూడిక;
}

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_reduce_కూడిక_i32(HVX_సదిశ లోపల) {
    తిరిగి hvx_vec_reduce_కూడిక_n_i32(లోపల, 32);
}

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_reduce_కూడిక_n_qf32(HVX_సదిశ లోపల, గుర్తులేని పూర్ణాంకం n) {
    గుర్తులేని పూర్ణాంకం మొత్తం = n * 4;  // total vec nbytes
    గుర్తులేని పూర్ణాంకం వెడల్పు = 4;      // fp32 nbytes

    HVX_సదిశ కూడిక = లోపల, కూడిక_t;
    అయితే (వెడల్పు < మొత్తం) {
        కూడిక_t = Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(కూడిక), వెడల్పు);  // rotate right
        కూడిక   = Q6_Vqf32_vadd_Vqf32Vsf(కూడిక, కూడిక_t);             // elementwise sum
        వెడల్పు = వెడల్పు << 1;
    }
    తిరిగి కూడిక;
}

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_reduce_కూడిక_qf32(HVX_సదిశ లోపల) {
    తిరిగి hvx_vec_reduce_కూడిక_n_qf32(లోపల, 32);
}

#ఒకవేళ __HVX_ARCH__ > 75

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_reduce_కూడిక_f32x4(HVX_సదిశ_x4 లోపల) {
    HVX_సదిశ_Pair కూడిక_p01 = Q6_W_vshuff_VVR(లోపల.v[1], లోపల.v[0], 4);
    HVX_సదిశ_Pair కూడిక_p23 = Q6_W_vshuff_VVR(లోపల.v[3], లోపల.v[2], 4);
    HVX_సదిశ  కూడిక_sf01  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(కూడిక_p01), Q6_V_hi_W(కూడిక_p01));
    HVX_సదిశ  కూడిక_sf23  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(కూడిక_p23), Q6_V_hi_W(కూడిక_p23));

    HVX_సదిశ_Pair కూడిక_p0123 = Q6_W_vshuff_VVR(కూడిక_sf23, కూడిక_sf01, 8);
    HVX_సదిశ  కూడిక_sf       = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(కూడిక_p0123), Q6_V_hi_W(కూడిక_p0123));

    కూడిక_sf = Q6_Vsf_vadd_VsfVsf(కూడిక_sf, Q6_V_vror_VR(కూడిక_sf, VLEN / 2));
    కూడిక_sf = Q6_Vsf_vadd_VsfVsf(కూడిక_sf, Q6_V_vror_VR(కూడిక_sf, VLEN / 4));
    కూడిక_sf = Q6_Vsf_vadd_VsfVsf(కూడిక_sf, Q6_V_vror_VR(కూడిక_sf, VLEN / 8));
    తిరిగి కూడిక_sf;
}

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_reduce_కూడిక_f32x2(HVX_సదిశ in0, HVX_సదిశ in1) {
    HVX_సదిశ_Pair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_సదిశ  కూడిక_sf  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    కూడిక_sf = Q6_Vsf_vadd_VsfVsf(కూడిక_sf, Q6_V_vror_VR(కూడిక_sf, VLEN / 2));
    కూడిక_sf = Q6_Vsf_vadd_VsfVsf(కూడిక_sf, Q6_V_vror_VR(కూడిక_sf, VLEN / 4));
    కూడిక_sf = Q6_Vsf_vadd_VsfVsf(కూడిక_sf, Q6_V_vror_VR(కూడిక_sf, VLEN / 8));
    కూడిక_sf = Q6_Vsf_vadd_VsfVsf(కూడిక_sf, Q6_V_vror_VR(కూడిక_sf, VLEN / 16));
    తిరిగి కూడిక_sf;
}

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_reduce_కూడిక_n_f32(HVX_సదిశ లోపల, గుర్తులేని పూర్ణాంకం n) {
    గుర్తులేని పూర్ణాంకం మొత్తం = n * 4;  // total vec nbytes
    గుర్తులేని పూర్ణాంకం వెడల్పు = 4;      // fp32 nbytes

    HVX_సదిశ కూడిక = లోపల, కూడిక_t;
    అయితే (వెడల్పు < మొత్తం) {
        కూడిక_t = Q6_V_vror_VR(కూడిక, వెడల్పు);       // rotate right
        కూడిక   = Q6_Vsf_vadd_VsfVsf(కూడిక, కూడిక_t); // elementwise sum
        వెడల్పు = వెడల్పు << 1;
    }
    తిరిగి కూడిక;
}

#లేకపోతే

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_reduce_కూడిక_f32x4(HVX_సదిశ_x4 లోపల) {
    HVX_సదిశ_Pair కూడిక_p01  = Q6_W_vshuff_VVR(లోపల.v[1], లోపల.v[0], 4);
    HVX_సదిశ_Pair కూడిక_p23  = Q6_W_vshuff_VVR(లోపల.v[3], లోపల.v[2], 4);
    HVX_సదిశ     కూడిక_qf01 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(కూడిక_p01), Q6_V_hi_W(కూడిక_p01));
    HVX_సదిశ     కూడిక_qf23 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(కూడిక_p23), Q6_V_hi_W(కూడిక_p23));

    HVX_సదిశ_Pair కూడిక_p0123 = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(కూడిక_qf23), Q6_Vsf_equals_Vqf32(కూడిక_qf01), 8);
    HVX_సదిశ     కూడిక_qf    = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(కూడిక_p0123), Q6_V_hi_W(కూడిక_p0123));

    కూడిక_qf = Q6_Vqf32_vadd_Vqf32Vsf(కూడిక_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(కూడిక_qf), VLEN / 2));
    కూడిక_qf = Q6_Vqf32_vadd_Vqf32Vsf(కూడిక_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(కూడిక_qf), VLEN / 4));
    కూడిక_qf = Q6_Vqf32_vadd_Vqf32Vsf(కూడిక_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(కూడిక_qf), VLEN / 8));
    తిరిగి Q6_Vsf_equals_Vqf32(కూడిక_qf);
}

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_reduce_కూడిక_f32x2(HVX_సదిశ in0, HVX_సదిశ in1) {
    HVX_సదిశ_Pair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_సదిశ  కూడిక_qf  = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    కూడిక_qf = Q6_Vqf32_vadd_Vqf32Vsf(కూడిక_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(కూడిక_qf), VLEN / 2));
    కూడిక_qf = Q6_Vqf32_vadd_Vqf32Vsf(కూడిక_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(కూడిక_qf), VLEN / 4));
    కూడిక_qf = Q6_Vqf32_vadd_Vqf32Vsf(కూడిక_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(కూడిక_qf), VLEN / 8));
    కూడిక_qf = Q6_Vqf32_vadd_Vqf32Vsf(కూడిక_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(కూడిక_qf), VLEN / 16));
    తిరిగి Q6_Vsf_equals_Vqf32(కూడిక_qf);
}

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_reduce_కూడిక_n_f32(HVX_సదిశ లోపల, గుర్తులేని పూర్ణాంకం n) {
    గుర్తులేని పూర్ణాంకం మొత్తం = n * 4;  // total vec nbytes
    గుర్తులేని పూర్ణాంకం వెడల్పు = 4;      // fp32 nbytes

    HVX_సదిశ కూడిక = లోపల, కూడిక_t;
    అయితే (వెడల్పు < మొత్తం) {
        కూడిక_t = Q6_V_vror_VR(కూడిక, వెడల్పు);                               // rotate right
        కూడిక   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(కూడిక, కూడిక_t));  // elementwise sum
        వెడల్పు = వెడల్పు << 1;
    }
    తిరిగి కూడిక;
}

#అయితే_ముగింపు

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_reduce_కూడిక_f32(HVX_సదిశ లోపల) {
    తిరిగి hvx_vec_reduce_కూడిక_n_f32(లోపల, 32);
}

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_reduce_max_f16(HVX_సదిశ లోపల) {
    గుర్తులేని మొత్తం = 128;  // total vec nbytes
    గుర్తులేని వెడల్పు = 2;    // fp16 nbytes

    HVX_సదిశ _max = లోపల, _max_t;
    అయితే (వెడల్పు < మొత్తం) {
        _max_t = Q6_V_vror_VR(_max, వెడల్పు);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        వెడల్పు  = వెడల్పు << 1;
    }

    తిరిగి _max;
}

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_reduce_max2_f16(HVX_సదిశ లోపల, HVX_సదిశ _max) {
    గుర్తులేని మొత్తం = 128;  // total vec nbytes
    గుర్తులేని వెడల్పు = 2;    // fp32 nbytes

    HVX_సదిశ _max_t;

    _max = Q6_Vhf_vmax_VhfVhf(లోపల, _max);
    అయితే (వెడల్పు < మొత్తం) {
        _max_t = Q6_V_vror_VR(_max, వెడల్పు);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        వెడల్పు  = వెడల్పు << 1;
    }

    తిరిగి _max;
}

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_reduce_max_f32(HVX_సదిశ లోపల) {
    గుర్తులేని మొత్తం = 128;  // total vec nbytes
    గుర్తులేని వెడల్పు = 4;    // fp32 nbytes

    HVX_సదిశ _max = లోపల, _max_t;
    అయితే (వెడల్పు < మొత్తం) {
        _max_t = Q6_V_vror_VR(_max, వెడల్పు);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        వెడల్పు  = వెడల్పు << 1;
    }

    తిరిగి _max;
}

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_reduce_max2_f32(HVX_సదిశ లోపల, HVX_సదిశ _max) {
    గుర్తులేని మొత్తం = 128;  // total vec nbytes
    గుర్తులేని వెడల్పు = 4;    // fp32 nbytes

    HVX_సదిశ _max_t;

    _max = Q6_Vsf_vmax_VsfVsf(లోపల, _max);
    అయితే (వెడల్పు < మొత్తం) {
        _max_t = Q6_V_vror_VR(_max, వెడల్పు);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        వెడల్పు  = వెడల్పు << 1;
    }

    తిరిగి _max;
}

#నిర్వచించు hvx_reduce_loop_body(src_type, init_vec, pad_vec, vec_op, reduce_op, scalar_reduce) \
    చేయి {                                                                                    \
        src_type * పరిమిత vsrc = (src_type *) src;                                        \
        HVX_సదిశ acc = init_vec;                                                          \
                                                                                            \
        స్థిరరాశి uint32_t elem_కొలత = పరిమాణం(దశాంశం);                                           \
        స్థిరరాశి uint32_t epv  = 128 / elem_కొలత;                                              \
        స్థిరరాశి uint32_t nvec = అంకె_elems / epv;                                              \
        స్థిరరాశి uint32_t nloe = అంకె_elems % epv;                                              \
                                                                                            \
        uint32_t i = 0;                                                                     \
        _Pragma("unroll(4)")                                                                \
        కొరకు (; i < nvec; i++) {                                                             \
            acc = vec_op(acc, vsrc[i]);                                                     \
        }                                                                                   \
        ఒకవేళ (nloe) {                                                                         \
            స్థిరరాశి దశాంశం * srcf = (స్థిరరాశి దశాంశం *) src + i * epv;                             \
            HVX_సదిశ లోపల = *(HVX_UVector *) srcf;                                          \
            HVX_సదిశ తాత్కాలిక = Q6_V_valign_VVR(లోపల, pad_vec, nloe * elem_కొలత);               \
            acc = vec_op(acc, తాత్కాలిక);                                                        \
        }                                                                                   \
        HVX_సదిశ v = reduce_op(acc);                                                      \
        తిరిగి scalar_reduce(v);                                                            \
    } అయితే(0)

#నిర్వచించు HVX_REDUCE_MAX_OP(acc, val) Q6_Vsf_vmax_VsfVsf(acc, val)
#నిర్వచించు HVX_REDUCE_కూడిక_OP(acc, val) Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(acc), val)
#నిర్వచించు HVX_కూడిక_SQ_OP(acc, val) Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(val, val))
#నిర్వచించు HVX_REDUCE_MAX_SCALAR(v) hvx_vec_get_f32(v)
#నిర్వచించు HVX_REDUCE_కూడిక_SCALAR(v) hvx_vec_get_f32(Q6_Vsf_equals_Vqf32(v))

// Max variants

స్థిర అంతఃవరుస దశాంశం hvx_reduce_max_f32_a(స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం అంకె_elems) {
    HVX_సదిశ init_vec = hvx_vec_splat_f32(((స్థిరరాశి దశాంశం *) src)[0]);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_reduce_loop_body(HVX_సదిశ, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

స్థిర అంతఃవరుస దశాంశం hvx_reduce_max_f32_u(స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం అంకె_elems) {
    HVX_సదిశ init_vec = hvx_vec_splat_f32(((స్థిరరాశి దశాంశం *) src)[0]);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

స్థిర అంతఃవరుస దశాంశం hvx_reduce_max_f32(స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం అంకె_elems) {
    ఒకవేళ (hex_is_aligned((శూన్య *) src, 128)) {
        తిరిగి hvx_reduce_max_f32_a(src, అంకె_elems);
    } లేకపోతే {
        తిరిగి hvx_reduce_max_f32_u(src, అంకె_elems);
    }
}

// Sum variants

స్థిర అంతఃవరుస దశాంశం hvx_reduce_కూడిక_f32_a(స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం అంకె_elems) {
    HVX_సదిశ init_vec = Q6_V_vsplat_R(0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_reduce_loop_body(HVX_సదిశ, init_vec, init_vec, HVX_REDUCE_కూడిక_OP, hvx_vec_reduce_కూడిక_qf32, HVX_REDUCE_కూడిక_SCALAR);
}

స్థిర అంతఃవరుస దశాంశం hvx_reduce_కూడిక_f32_u(స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం అంకె_elems) {
    HVX_సదిశ init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_కూడిక_OP, hvx_vec_reduce_కూడిక_qf32, HVX_REDUCE_కూడిక_SCALAR);
}

స్థిర అంతఃవరుస దశాంశం hvx_reduce_కూడిక_f32(స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం అంకె_elems) {
    ఒకవేళ (hex_is_aligned((శూన్య *) src, 128)) {
        తిరిగి hvx_reduce_కూడిక_f32_a(src, అంకె_elems);
    } లేకపోతే {
        తిరిగి hvx_reduce_కూడిక_f32_u(src, అంకె_elems);
    }
}

// Sum of squares variants

స్థిర అంతఃవరుస దశాంశం hvx_కూడిక_యొక్క_squares_f32_a(స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం అంకె_elems) {
    HVX_సదిశ init_vec = Q6_V_vsplat_R(0);
    ప్రతిజ్ఞా((uintptr_t) src % 128 == 0);
    hvx_reduce_loop_body(HVX_సదిశ, init_vec, init_vec, HVX_కూడిక_SQ_OP, hvx_vec_reduce_కూడిక_qf32, HVX_REDUCE_కూడిక_SCALAR);
}

స్థిర అంతఃవరుస దశాంశం hvx_కూడిక_యొక్క_squares_f32_u(స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం అంకె_elems) {
    HVX_సదిశ init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_కూడిక_SQ_OP, hvx_vec_reduce_కూడిక_qf32, HVX_REDUCE_కూడిక_SCALAR);
}

స్థిర అంతఃవరుస దశాంశం hvx_కూడిక_యొక్క_squares_f32(స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం అంకె_elems) {
    ఒకవేళ (hex_is_aligned((శూన్య *) src, 128)) {
        తిరిగి hvx_కూడిక_యొక్క_squares_f32_a(src, అంకె_elems);
    } లేకపోతే {
        తిరిగి hvx_కూడిక_యొక్క_squares_f32_u(src, అంకె_elems);
    }
}

// Signed 32-bit Integer Max variants

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_reduce_max_n_i32(HVX_సదిశ లోపల, గుర్తులేని పూర్ణాంకం n) {
    గుర్తులేని పూర్ణాంకం మొత్తం = n * 4;  // total vec nbytes
    గుర్తులేని పూర్ణాంకం వెడల్పు = 4;      // int32 nbytes

    HVX_సదిశ max_val = లోపల, max_t;
    అయితే (వెడల్పు < మొత్తం) {
        max_t = Q6_V_vror_VR(max_val, వెడల్పు);          // rotate right
        max_val = Q6_Vw_vmax_VwVw(max_t, max_val);      // elementwise signed max
        వెడల్పు = వెడల్పు << 1;
    }
    తిరిగి max_val;
}

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_reduce_max_i32(HVX_సదిశ లోపల) {
    తిరిగి hvx_vec_reduce_max_n_i32(లోపల, 32);
}

స్థిర అంతఃవరుస int32_t hvx_reduce_max_i32_a(స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం అంకె_elems) {
    HVX_సదిశ init_vec = Q6_V_vsplat_R(((స్థిరరాశి int32_t *) src)[0]);
    HVX_సదిశ pad_vec  = Q6_V_vsplat_R(0x80000000);
    ప్రతిజ్ఞా((uintptr_t) src % 128 == 0);
    hvx_reduce_loop_body(HVX_సదిశ, init_vec, pad_vec, Q6_Vw_vmax_VwVw, hvx_vec_reduce_max_i32, hvx_vec_get_i32);
}

స్థిర అంతఃవరుస int32_t hvx_reduce_max_i32_u(స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం అంకె_elems) {
    HVX_సదిశ init_vec = Q6_V_vsplat_R(((స్థిరరాశి int32_t *) src)[0]);
    HVX_సదిశ pad_vec  = Q6_V_vsplat_R(0x80000000);
    hvx_reduce_loop_body(HVX_UVector, init_vec, pad_vec, Q6_Vw_vmax_VwVw, hvx_vec_reduce_max_i32, hvx_vec_get_i32);
}

స్థిర అంతఃవరుస int32_t hvx_reduce_max_i32(స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం అంకె_elems) {
    ఒకవేళ (hex_is_aligned((శూన్య *) src, 128)) {
        తిరిగి hvx_reduce_max_i32_a(src, అంకె_elems);
    } లేకపోతే {
        తిరిగి hvx_reduce_max_i32_u(src, అంకె_elems);
    }
}

స్థిర అంతఃవరుస శూన్య hvx_argmax_f32(
    స్థిరరాశి దశాంశం * పరిమిత src,
    uint32_t n,
    uint32_t స్థానభ్రంశం,
    దశాంశం * బహిర్గతం_val,
    int32_t * బహిర్గతం_idx
) {
    ఒకవేళ (n == 0) {
        *బహిర్గతం_val = -INFINITY;
        *బహిర్గతం_idx = (int32_t) స్థానభ్రంశం;
        తిరిగి;
    }

    ఒకవేళ (n < 32 || !hex_is_aligned((శూన్య *) src, 128)) {
        దశాంశం ఉత్తమ_val = src[0];
        int32_t ఉత్తమ_idx = (int32_t) స్థానభ్రంశం;
        కొరకు (uint32_t i = 1; i < n; i++) {
            ఒకవేళ (src[i] > ఉత్తమ_val) {
                ఉత్తమ_val = src[i];
                ఉత్తమ_idx = (int32_t) (స్థానభ్రంశం + i);
            }
        }
        *బహిర్గతం_val = ఉత్తమ_val;
        *బహిర్గతం_idx = ఉత్తమ_idx;
        తిరిగి;
    }

    స్థిర స్థిరరాశి int32_t c_lane_idx[32] __attribute__((aligned(128))) = {
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    };

    స్థిరరాశి HVX_సదిశ v_init_idx = *(స్థిరరాశి HVX_సదిశ *) c_lane_idx;
    స్థిరరాశి HVX_సదిశ v_అడుగు     = Q6_V_vsplat_R(32);
    HVX_సదిశ v_cur_idx        = Q6_Vw_vadd_VwVw(v_init_idx, Q6_V_vsplat_R((int32_t) స్థానభ్రంశం));
    HVX_సదిశ v_max_val        = hvx_vec_splat_f32(-INFINITY);
    HVX_సదిశ v_max_idx        = v_cur_idx;

    స్థిరరాశి HVX_సదిశ * vsrc = (స్థిరరాశి HVX_సదిశ *) src;
    స్థిరరాశి uint32_t nvec = n / 32;

    కొరకు (uint32_t vi = 0; vi < nvec; vi++) {
        HVX_సదిశ v = vsrc[vi];
        HVX_సదిశ_Pred pred = Q6_Q_vcmp_gt_VsfVsf(v, v_max_val);
        v_max_val = Q6_V_vmux_QVV(pred, v, v_max_val);
        v_max_idx = Q6_V_vmux_QVV(pred, v_cur_idx, v_max_idx);
        v_cur_idx = Q6_Vw_vadd_VwVw(v_cur_idx, v_అడుగు);
    }

    HVX_సదిశ_Alias u_val, u_idx;
    u_val.v = v_max_val;
    u_idx.v = v_max_idx;

    దశాంశం ఉత్తమ_val = u_val.fp32[0];
    int32_t ఉత్తమ_idx = (int32_t) u_idx.w[0];
    కొరకు (పూర్ణాంకం i = 1; i < 32; i++) {
        ఒకవేళ (u_val.fp32[i] > ఉత్తమ_val) {
            ఉత్తమ_val = u_val.fp32[i];
            ఉత్తమ_idx = (int32_t) u_idx.w[i];
        }
    }

    కొరకు (uint32_t i = nvec * 32; i < n; i++) {
        ఒకవేళ (src[i] > ఉత్తమ_val) {
            ఉత్తమ_val = src[i];
            ఉత్తమ_idx = (int32_t) (స్థానభ్రంశం + i);
        }
    }

    *బహిర్గతం_val = ఉత్తమ_val;
    *బహిర్గతం_idx = ఉత్తమ_idx;
}

#నిర్వచనం_రద్దు hvx_reduce_loop_body
#నిర్వచనం_రద్దు HVX_REDUCE_MAX_OP
#నిర్వచనం_రద్దు HVX_REDUCE_కూడిక_OP
#నిర్వచనం_రద్దు HVX_REDUCE_MAX_SCALAR
#నిర్వచనం_రద్దు HVX_REDUCE_కూడిక_SCALAR
#నిర్వచనం_రద్దు HVX_కూడిక_SQ_OP

#అయితే_ముగింపు /* HVX_REDUCE_H */
