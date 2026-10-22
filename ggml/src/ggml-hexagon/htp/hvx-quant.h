#اگر_غیر_تعریف_شدہ HVX_QUANT_H
#تعریف_کرو HVX_QUANT_H

#شامل <math.h>
#شامل <stdint.h>
#شامل <string.h>

#شامل "htp-tensor.h"
#شامل "hvx-arith.h"
#شامل "hvx-base.h"
#شامل "hvx-reduce.h"
#شامل "hvx-repl.h"
#شامل "hvx-utils.h"

#اگر_غیر_تعریف_شدہ GGML_COMMON_DECL_C
#تعریف_کرو GGML_COMMON_DECL_C
#اگر_ختم
#شامل "ggml-common.h"
#شامل "ggml-impl.h"

ساکن اندرونی خالی hvx_quantize_صف_q8_0_f32(خالی * محدود dst_ptr, ثابت اعشاری * محدود src_ptr, صحیح n) {
    ثابت صحیح nb = n / QK8_0;
    بلاک_q8_0 * dst = (بلاک_q8_0 *) dst_ptr;
    HVX_سمتیہ zero = Q6_V_vzero();

    صحیح i = 0;
    برائے (; i + 3 < nb; i += 4) {
        HVX_سمتیہ * vx = (HVX_سمتیہ *) (src_ptr + i * QK8_0);

        HVX_سمتیہ vmax0_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[0]));
        HVX_سمتیہ vmax1_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[1]));
        HVX_سمتیہ vmax2_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[2]));
        HVX_سمتیہ vmax3_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[3]));

        HVX_سمتیہ vx0_qf = Q6_Vqf32_vsub_VsfVsf(vx[0], zero);
        HVX_سمتیہ vx1_qf = Q6_Vqf32_vsub_VsfVsf(vx[1], zero);
        HVX_سمتیہ vx2_qf = Q6_Vqf32_vsub_VsfVsf(vx[2], zero);
        HVX_سمتیہ vx3_qf = Q6_Vqf32_vsub_VsfVsf(vx[3], zero);

        HVX_سمتیہ vmax0_qf = Q6_Vqf32_vsub_VsfVsf(vmax0_sf, zero);
        HVX_سمتیہ vmax1_qf = Q6_Vqf32_vsub_VsfVsf(vmax1_sf, zero);
        HVX_سمتیہ vmax2_qf = Q6_Vqf32_vsub_VsfVsf(vmax2_sf, zero);
        HVX_سمتیہ vmax3_qf = Q6_Vqf32_vsub_VsfVsf(vmax3_sf, zero);

        HVX_سمتیہ vmax01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax1_qf, vmax0_qf)));
        HVX_سمتیہ vmax23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax3_qf, vmax2_qf)));

        HVX_سمتیہ vx01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx1_qf, vx0_qf)));
        HVX_سمتیہ vx23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx3_qf, vx2_qf)));

        HVX_سمتیہ vd01_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax01_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
        HVX_سمتیہ vd23_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax23_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
        HVX_سمتیہ vd01_hf   = Q6_Vhf_equals_Vqf16(vd01_qf16);
        HVX_سمتیہ vd23_hf   = Q6_Vhf_equals_Vqf16(vd23_qf16);

        HVX_سمتیہ vd01_inv_hf = hvx_vec_inverse_f16(vd01_hf);
        HVX_سمتیہ vd23_inv_hf = hvx_vec_inverse_f16(vd23_hf);
        vx01_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx01_hf, vd01_inv_hf));
        vx23_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx23_hf, vd23_inv_hf));

        HVX_سمتیہ vx01_i16 = hvx_vec_i16_from_hf_rnd_sat(vx01_hf);
        HVX_سمتیہ vx23_i16 = hvx_vec_i16_from_hf_rnd_sat(vx23_hf);
        HVX_سمتیہ vx_i8    = Q6_Vb_vpack_VhVh_sat(vx23_i16, vx01_i16);

        hvx_vec_store_u(&dst[i + 0].d, 2, vd01_hf);
        hvx_vec_store_u(dst[i + 0].qs, 32, vx_i8);

        hvx_vec_store_u(&dst[i + 1].d, 2, Q6_V_vror_VR(vd01_hf, 64));
        hvx_vec_store_u(dst[i + 1].qs, 32, Q6_V_vror_VR(vx_i8, 32));

        hvx_vec_store_u(&dst[i + 2].d, 2, vd23_hf);
        hvx_vec_store_u(dst[i + 2].qs, 32, Q6_V_vror_VR(vx_i8, 64));

        hvx_vec_store_u(&dst[i + 3].d, 2, Q6_V_vror_VR(vd23_hf, 64));
        hvx_vec_store_u(dst[i + 3].qs, 32, Q6_V_vror_VR(vx_i8, 96));
    }

    برائے (; i < nb; i++) {
        ثابت اعشاری * بلاک_src = src_ptr + i * QK8_0;
        HVX_سمتیہ vx = *(ثابت HVX_UVector *) بلاک_src;
        HVX_سمتیہ v_abs = hvx_vec_abs_f32(vx);
        HVX_سمتیہ v_max = hvx_vec_reduce_max_f32(v_abs);
        اعشاری amax = hvx_vec_get_f32(v_max);

        ثابت اعشاری d = amax / 127.0f;
        ثابت اعشاری id = d ? (1.0f / d) : 0.0f;
        dst[i].d = GGML_FP32_مقصد_FP16(d);

        HVX_سمتیہ vid = hvx_vec_splat_f32(id);
        HVX_سمتیہ v_scaled = hvx_vec_mul_f32_f32(vx, vid);
        HVX_سمتیہ v_scaled_qf = Q6_Vqf32_vsub_VsfVsf(v_scaled, zero);
        HVX_سمتیہ v_scaled_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(zero, v_scaled_qf)));
        HVX_سمتیہ v_i16 = hvx_vec_i16_from_hf_rnd_sat(v_scaled_hf);
        HVX_سمتیہ v_i8  = Q6_Vb_vpack_VhVh_sat(zero, v_i16);

        hvx_vec_store_u(dst[i].qs, 32, v_i8);
    }
}

ساکن اندرونی خالی hvx_dequantize_صف_flat_q8_0_f32(اعشاری * محدود dst_ptr, ثابت خالی * محدود src_ptr, uint32_t n) {
    ثابت صحیح nb = n / QK8_0;
    ثابت بلاک_q8_0 * src = (ثابت بلاک_q8_0 *) src_ptr;

    برائے (صحیح i = 0; i < nb; i++) {
        HVX_سمتیہ vd_f16     = Q6_Vh_vsplat_R(*(ثابت int16_t *) &src[i].d);
        HVX_سمتیہ_Pair vp_f32 = hvx_vec_f16_مقصد_f32(vd_f16);
        HVX_سمتیہ vd         = Q6_V_lo_W(vp_f32);

        HVX_سمتیہ vq_i8 = *(ثابت HVX_UVector *) src[i].qs;

        HVX_سمتیہ_Pair p16   = Q6_Wh_vunpack_Vb(vq_i8);
        HVX_سمتیہ     v_i16 = Q6_V_lo_W(p16);
        HVX_سمتیہ_Pair p32   = Q6_Ww_vunpack_Vh(v_i16);
        HVX_سمتیہ     v_i32 = Q6_V_lo_W(p32);

        HVX_سمتیہ v_f32 = Q6_Vsf_equals_Vw(v_i32);
        HVX_سمتیہ res   = hvx_vec_mul_f32_f32(v_f32, vd);

        اعشاری * بلاک_dst = dst_ptr + i * QK8_0;
        hvx_vmem(بلاک_dst) = res;
    }
}

ساکن اندرونی خالی hvx_dequantize_صف_flat_q8_0_f16(__fp16 * محدود dst_ptr, ثابت خالی * محدود src_ptr, uint32_t n) {
    ثابت صحیح nb = n / QK8_0;
    ثابت بلاک_q8_0 * src = (ثابت بلاک_q8_0 *) src_ptr;

    برائے (صحیح i = nb - 1; i >= 0; i--) {
        HVX_سمتیہ vd_f16     = Q6_Vh_vsplat_R(*(ثابت int16_t *) &src[i].d);
        HVX_سمتیہ_Pair vp_f32 = hvx_vec_f16_مقصد_f32(vd_f16);
        HVX_سمتیہ vd         = Q6_V_lo_W(vp_f32);

        HVX_سمتیہ vq_i8 = *(ثابت HVX_UVector *) src[i].qs;

        HVX_سمتیہ_Pair p16   = Q6_Wh_vunpack_Vb(vq_i8);
        HVX_سمتیہ     v_i16 = Q6_V_lo_W(p16);
        HVX_سمتیہ_Pair p32   = Q6_Ww_vunpack_Vh(v_i16);
        HVX_سمتیہ     v_i32 = Q6_V_lo_W(p32);

        HVX_سمتیہ v_f32 = Q6_Vsf_equals_Vw(v_i32);
        HVX_سمتیہ res_f32 = hvx_vec_mul_f32_f32(v_f32, vd);

        HVX_سمتیہ res_f16 = hvx_vec_f32_مقصد_f16(res_f32, Q6_V_vzero());

        __fp16 * بلاک_dst = dst_ptr + i * QK8_0;
        hvx_vec_store_u(بلاک_dst, QK8_0 * حجم(__fp16), res_f16);
    }
}

ساکن اندرونی خالی hvx_dequantize_صف_flat_f16_f32(اعشاری * محدود dst_ptr, ثابت خالی * محدود src_ptr, uint32_t n) {
    ثابت uint32_t nvec = n / VLEN_FP32;
    ثابت uint32_t nloe = n % VLEN_FP32;
    ثابت _Float16 * src = (ثابت _Float16 *) src_ptr;

    برائے (uint32_t i = 0; i < nvec; i++) {
        HVX_سمتیہ v_f16 = *(ثابت HVX_UVector *) (src + i * VLEN_FP32);
        HVX_سمتیہ_Pair vp_f32 = hvx_vec_f16_مقصد_f32(v_f16);
        HVX_سمتیہ res = Q6_V_lo_W(vp_f32);

        hvx_vmem(dst_ptr + i * VLEN_FP32) = res;
    }

    اگر (nloe) {
        HVX_سمتیہ v_f16 = *(ثابت HVX_UVector *) (src + nvec * VLEN_FP32);
        HVX_سمتیہ_Pair vp_f32 = hvx_vec_f16_مقصد_f32(v_f16);
        HVX_سمتیہ res = Q6_V_lo_W(vp_f32);

        hvx_vec_store_a(dst_ptr + nvec * VLEN_FP32, nloe * حجم(اعشاری), res);
    }
}

ساکن اندرونی خالی hvx_dequantize_صف_flat_bf16_f32(اعشاری * محدود dst_ptr, ثابت خالی * محدود src_ptr, uint32_t n) {
    ثابت uint32_t nvec = n / VLEN_FP32;
    ثابت uint32_t nloe = n % VLEN_FP32;
    ثابت uint16_t * src = (ثابت uint16_t *) src_ptr;

    برائے (uint32_t i = 0; i < nvec; i++) {
        HVX_سمتیہ v_bf16 = *(ثابت HVX_UVector *) (src + i * VLEN_FP32);
        HVX_سمتیہ_Pair p = Q6_Ww_vunpack_Vh(v_bf16);
        HVX_سمتیہ res = Q6_Vw_vasl_VwR(Q6_V_lo_W(p), 16);

        hvx_vmem(dst_ptr + i * VLEN_FP32) = res;
    }

    اگر (nloe) {
        HVX_سمتیہ v_bf16 = *(ثابت HVX_UVector *) (src + nvec * VLEN_FP32);
        HVX_سمتیہ_Pair p = Q6_Ww_vunpack_Vh(v_bf16);
        HVX_سمتیہ res = Q6_Vw_vasl_VwR(Q6_V_lo_W(p), 16);

        hvx_vec_store_a(dst_ptr + nvec * VLEN_FP32, nloe * حجم(اعشاری), res);
    }
}

ساکن اندرونی خالی hvx_dequantize_صف_flat_q1_0_f32(اعشاری * محدود dst_ptr, ثابت خالی * محدود src_ptr, uint32_t n) {
    ساکن ثابت uint32_t __attribute__((aligned(128))) ثنائی_ہندسہ_نقاب[32] = {
        1u <<  0, 1u <<  1, 1u <<  2, 1u <<  3, 1u <<  4, 1u <<  5, 1u <<  6, 1u <<  7,
        1u <<  8, 1u <<  9, 1u << 10, 1u << 11, 1u << 12, 1u << 13, 1u << 14, 1u << 15,
        1u << 16, 1u << 17, 1u << 18, 1u << 19, 1u << 20, 1u << 21, 1u << 22, 1u << 23,
        1u << 24, 1u << 25, 1u << 26, 1u << 27, 1u << 28, 1u << 29, 1u << 30, 1u << 31,
    };
    ثابت صحیح nb = n / QK1_0;
    ثابت بلاک_q1_0 * src = (ثابت بلاک_q1_0 *) src_ptr;
    ثابت HVX_سمتیہ v_ثنائی_ہندسہ_نقاب = *(ثابت HVX_سمتیہ *) ثنائی_ہندسہ_نقاب;
    ثابت HVX_سمتیہ v_zero     = Q6_V_vzero();
    ثابت HVX_سمتیہ v_sign_ثنائی_ہندسہ = Q6_V_vsplat_R(0x80000000);

    برائے (صحیح i = 0; i < nb; i++) {
        HVX_سمتیہ vd_f16 = Q6_Vh_vsplat_R(*(ثابت int16_t *) &src[i].d);
        HVX_سمتیہ vd     = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(vd_f16));
        HVX_سمتیہ neg_vd = Q6_V_vxor_VV(vd, v_sign_ثنائی_ہندسہ);
        HVX_سمتیہ v_qs   = *(ثابت HVX_UVector *) src[i].qs;

        اعشاری * بلاک_dst = dst_ptr + i * QK1_0;
        #ہدایت unroll
        برائے (صحیح q = 0; q < 4; q++) {
            HVX_سمتیہ v_بٹ = hvx_vec_repl_u32(q == 0 ? v_qs : Q6_V_vror_VR(v_qs, 4 * q));
            HVX_سمتیہ_Pred q_zero = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(v_بٹ, v_ثنائی_ہندسہ_نقاب), v_zero);
            hvx_vmem(بلاک_dst + q * 32) = Q6_V_vmux_QVV(q_zero, neg_vd, vd);
        }
    }
}

ساکن اندرونی خالی hvx_dequantize_صف_flat_q2_0_f32(اعشاری * محدود dst_ptr, ثابت خالی * محدود src_ptr, uint32_t n) {
    ثابت صحیح nb = n / QK2_0;
    ثابت بلاک_q2_0 * src = (ثابت بلاک_q2_0 *) src_ptr;
    ثابت HVX_سمتیہ نقاب_03 = Q6_Vb_vsplat_R(0x03);
    ثابت HVX_سمتیہ v_one   = Q6_Vb_vsplat_R(1);

    برائے (صحیح i = 0; i < nb; i++) {
        HVX_سمتیہ vd_f16 = Q6_Vh_vsplat_R(*(ثابت int16_t *) &src[i].d);
        HVX_سمتیہ vd     = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(vd_f16));

        HVX_سمتیہ vq  = *(ثابت HVX_UVector *) src[i].qs;
        HVX_سمتیہ v0  = Q6_V_vand_VV(vq, نقاب_03);
        HVX_سمتیہ v1  = Q6_V_vand_VV(Q6_Vub_vlsr_VubR(vq, 2), نقاب_03);
        HVX_سمتیہ v2  = Q6_V_vand_VV(Q6_Vub_vlsr_VubR(vq, 4), نقاب_03);
        HVX_سمتیہ v3  = Q6_Vub_vlsr_VubR(vq, 6);

        HVX_سمتیہ v01 = Q6_V_lo_W(Q6_W_vshuff_VVR(v1, v0, -1));
        HVX_سمتیہ v23 = Q6_V_lo_W(Q6_W_vshuff_VVR(v3, v2, -1));
        HVX_سمتیہ v64 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(Q6_W_vshuff_VVR(v23, v01, -2)), v_one);

        HVX_سمتیہ_Pair p16 = Q6_Wh_vunpack_Vb(v64);
        HVX_سمتیہ_Pair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));

        اعشاری * بلاک_dst = dst_ptr + i * QK2_0;
        hvx_vmem(بلاک_dst + 0)  = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), vd);
        hvx_vmem(بلاک_dst + 32) = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_hi_W(p32)), vd);
    }
}

ساکن اندرونی خالی hvx_dequantize_صف_flat_q4_0_f32(اعشاری * محدود dst_ptr, ثابت خالی * محدود src_ptr, uint32_t n) {
    ثابت صحیح nb = n / QK4_0;
    ثابت بلاک_q4_0 * src = (ثابت بلاک_q4_0 *) src_ptr;

    ثابت HVX_سمتیہ_Pred q16 = Q6_Q_vsetq_R(16);
    ثابت HVX_سمتیہ v_eight = Q6_Vb_vsplat_R(8);
    ثابت HVX_سمتیہ v_zero  = Q6_V_vzero();

    برائے (صحیح i = 0; i < nb; i++) {
        HVX_سمتیہ vd_f16     = Q6_Vh_vsplat_R(*(ثابت int16_t *) &src[i].d);
        HVX_سمتیہ_Pair vp_f32 = hvx_vec_f16_مقصد_f32(vd_f16);
        HVX_سمتیہ vd         = Q6_V_lo_W(vp_f32);

        HVX_سمتیہ vq = *(ثابت HVX_UVector *) src[i].qs;
        HVX_سمتیہ v_lo = Q6_V_vand_VV(vq, Q6_Vb_vsplat_R(0x0F));
        HVX_سمتیہ v_hi = Q6_Vub_vlsr_VubR(vq, 4);

        HVX_سمتیہ v_lo_s8 = Q6_Vb_vsub_VbVb(v_lo, v_eight);
        HVX_سمتیہ v_hi_s8 = Q6_Vb_vsub_VbVb(v_hi, v_eight);

        HVX_سمتیہ v32 = Q6_V_vmux_QVV(q16, v_lo_s8, Q6_V_vlalign_VVR(v_hi_s8, v_zero, 16));

        HVX_سمتیہ_Pair p16   = Q6_Wh_vunpack_Vb(v32);
        HVX_سمتیہ     v_i16 = Q6_V_lo_W(p16);
        HVX_سمتیہ_Pair p32   = Q6_Ww_vunpack_Vh(v_i16);
        HVX_سمتیہ     v_i32 = Q6_V_lo_W(p32);

        HVX_سمتیہ v_f32 = Q6_Vsf_equals_Vw(v_i32);
        HVX_سمتیہ res   = hvx_vec_mul_f32_f32(v_f32, vd);

        hvx_vmem(dst_ptr + i * QK4_0) = res;
    }
}

ساکن اندرونی خالی hvx_dequantize_صف_flat_q4_k_f32(اعشاری * محدود dst_ptr, ثابت خالی * محدود src_ptr, uint32_t n) {
    ثابت صحیح nb = n / QK_K;
    ثابت بلاک_q4_K * src = (ثابت بلاک_q4_K *) src_ptr;

    برائے (صحیح i = 0; i < nb; i++) {
        ثابت اعشاری d   = GGML_FP16_مقصد_FP32(src[i].d);
        ثابت اعشاری min = GGML_FP16_مقصد_FP32(src[i].dmin);
        ثابت uint8_t * sc = src[i].scales;

        اعشاری * y = dst_ptr + i * QK_K;

        برائے (صحیح قدم = 0; قدم < 4; ++قدم) {
            uint8_t sc1, m1_u, sc2, m2_u;
            اگر (قدم < 2) {
                sc1  = sc[2 * قدم + 0] & 63;
                m1_u = sc[2 * قدم + 4] & 63;
                sc2  = sc[2 * قدم + 1] & 63;
                m2_u = sc[2 * قدم + 5] & 63;
            } ورنہ {
                ثابت صحیح s = قدم - 2;
                sc1  = (sc[8 + 2 * s + 0] & 0xF) | ((sc[2 * s + 0] >> 6) << 4);
                m1_u = (sc[8 + 2 * s + 0] >>  4) | ((sc[4 + 2 * s + 0] >> 6) << 4);
                sc2  = (sc[8 + 2 * s + 1] & 0xF) | ((sc[2 * s + 1] >> 6) << 4);
                m2_u = (sc[8 + 2 * s + 1] >>  4) | ((sc[4 + 2 * s + 1] >> 6) << 4);
            }

            HVX_سمتیہ vq = *(ثابت HVX_UVector *) (src[i].qs + قدم * 32);
            HVX_سمتیہ v_lo = Q6_V_vand_VV(vq, Q6_Vb_vsplat_R(0x0F));
            HVX_سمتیہ v_hi = Q6_Vub_vlsr_VubR(vq, 4);

            HVX_سمتیہ_Pair p16_lo = Q6_Wuh_vunpack_Vub(v_lo);
            HVX_سمتیہ_Pair p32_lo = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_lo));
            HVX_سمتیہ     v_i32_lo = Q6_V_lo_W(p32_lo);
            HVX_سمتیہ     v_f32_lo = Q6_Vsf_equals_Vw(v_i32_lo);

            HVX_سمتیہ vd1 = hvx_vec_splat_f32(d * sc1);
            HVX_سمتیہ vm1 = hvx_vec_splat_f32(min * m1_u);
            HVX_سمتیہ v_out1 = hvx_vec_sub_f32_f32(hvx_vec_mul_f32_f32(v_f32_lo, vd1), vm1);
            hvx_vmem(y + 0) = v_out1;

            HVX_سمتیہ_Pair p16_hi = Q6_Wuh_vunpack_Vub(v_hi);
            HVX_سمتیہ_Pair p32_hi = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_hi));
            HVX_سمتیہ     v_i32_hi = Q6_V_lo_W(p32_hi);
            HVX_سمتیہ     v_f32_hi = Q6_Vsf_equals_Vw(v_i32_hi);

            HVX_سمتیہ vd2 = hvx_vec_splat_f32(d * sc2);
            HVX_سمتیہ vm2 = hvx_vec_splat_f32(min * m2_u);
            HVX_سمتیہ v_out2 = hvx_vec_sub_f32_f32(hvx_vec_mul_f32_f32(v_f32_hi, vd2), vm2);
            hvx_vmem(y + 32) = v_out2;

            y += 64;
        }
    }
}

ساکن اندرونی خالی hvx_dequantize_صف_flat_q5_k_f32(اعشاری * محدود dst_ptr, ثابت خالی * محدود src_ptr, uint32_t n) {
    ثابت صحیح nb = n / QK_K;
    ثابت بلاک_q5_K * src = (ثابت بلاک_q5_K *) src_ptr;

    برائے (صحیح i = 0; i < nb; i++) {
        ثابت اعشاری d   = GGML_FP16_مقصد_FP32(src[i].d);
        ثابت اعشاری min = GGML_FP16_مقصد_FP32(src[i].dmin);
        ثابت uint8_t * sc = src[i].scales;

        HVX_سمتیہ v_qh = *(ثابت HVX_UVector *) src[i].qh;
        اعشاری * y = dst_ptr + i * QK_K;

        برائے (صحیح قدم = 0; قدم < 4; ++قدم) {
            uint8_t sc1, m1_u, sc2, m2_u;
            اگر (قدم < 2) {
                sc1  = sc[2 * قدم + 0] & 63;
                m1_u = sc[2 * قدم + 4] & 63;
                sc2  = sc[2 * قدم + 1] & 63;
                m2_u = sc[2 * قدم + 5] & 63;
            } ورنہ {
                ثابت صحیح s = قدم - 2;
                sc1  = (sc[8 + 2 * s + 0] & 0xF) | ((sc[2 * s + 0] >> 6) << 4);
                m1_u = (sc[8 + 2 * s + 0] >>  4) | ((sc[4 + 2 * s + 0] >> 6) << 4);
                sc2  = (sc[8 + 2 * s + 1] & 0xF) | ((sc[2 * s + 1] >> 6) << 4);
                m2_u = (sc[8 + 2 * s + 1] >>  4) | ((sc[4 + 2 * s + 1] >> 6) << 4);
            }

            HVX_سمتیہ v_ql = *(ثابت HVX_UVector *) (src[i].qs + قدم * 32);
            HVX_سمتیہ v_lo = Q6_V_vand_VV(v_ql, Q6_Vb_vsplat_R(0x0F));
            HVX_سمتیہ v_hi = Q6_Vub_vlsr_VubR(v_ql, 4);

            HVX_سمتیہ v_h1 = Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_qh, 2 * قدم), Q6_Vb_vsplat_R(1)), 4);
            HVX_سمتیہ v_val1 = Q6_V_vor_VV(v_lo, v_h1);

            HVX_سمتیہ_Pair p16_1 = Q6_Wuh_vunpack_Vub(v_val1);
            HVX_سمتیہ_Pair p32_1 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_1));
            HVX_سمتیہ     v_i32_1 = Q6_V_lo_W(p32_1);
            HVX_سمتیہ     v_f32_1 = Q6_Vsf_equals_Vw(v_i32_1);

            HVX_سمتیہ vd1 = hvx_vec_splat_f32(d * sc1);
            HVX_سمتیہ vm1 = hvx_vec_splat_f32(min * m1_u);
            HVX_سمتیہ v_out1 = hvx_vec_sub_f32_f32(hvx_vec_mul_f32_f32(v_f32_1, vd1), vm1);
            hvx_vmem(y + 0) = v_out1;

            HVX_سمتیہ v_h2 = Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_qh, 2 * قدم + 1), Q6_Vb_vsplat_R(1)), 4);
            HVX_سمتیہ v_val2 = Q6_V_vor_VV(v_hi, v_h2);

            HVX_سمتیہ_Pair p16_2 = Q6_Wuh_vunpack_Vub(v_val2);
            HVX_سمتیہ_Pair p32_2 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_2));
            HVX_سمتیہ     v_i32_2 = Q6_V_lo_W(p32_2);
            HVX_سمتیہ     v_f32_2 = Q6_Vsf_equals_Vw(v_i32_2);

            HVX_سمتیہ vd2 = hvx_vec_splat_f32(d * sc2);
            HVX_سمتیہ vm2 = hvx_vec_splat_f32(min * m2_u);
            HVX_سمتیہ v_out2 = hvx_vec_sub_f32_f32(hvx_vec_mul_f32_f32(v_f32_2, vd2), vm2);
            hvx_vmem(y + 32) = v_out2;

            y += 64;
        }
    }
}

ساکن اندرونی خالی hvx_dequantize_صف_flat_q6_k_f32(اعشاری * محدود dst_ptr, ثابت خالی * محدود src_ptr, uint32_t n) {
    ثابت صحیح nb = n / QK_K;
    ثابت بلاک_q6_K * src = (ثابت بلاک_q6_K *) src_ptr;

    ثابت HVX_سمتیہ_Pred q64 = Q6_Q_vsetq_R(64);
    ثابت HVX_سمتیہ v32_b   = Q6_Vb_vsplat_R(32);
    ثابت HVX_سمتیہ نقاب_0f = Q6_Vb_vsplat_R(0x0F);
    ثابت HVX_سمتیہ نقاب_03 = Q6_Vb_vsplat_R(0x03);

    برائے (صحیح i = 0; i < nb; i++) {
        ثابت اعشاری d = GGML_FP16_مقصد_FP32(src[i].d);

        ثابت uint8_t * محدود ql = src[i].ql;
        ثابت uint8_t * محدود qh = src[i].qh;
        ثابت int8_t  * محدود sc = src[i].scales;

        اعشاری * y = dst_ptr + i * QK_K;

        برائے (صحیح half = 0; half < 2; ++half) {
            اعشاری s[8];
            برائے (صحیح j = 0; j < 8; ++j) {
                s[j] = d * sc[j];
            }

            HVX_سمتیہ vscale1 = Q6_V_vmux_QVV(q64, hvx_vec_splat_f32(s[0]), hvx_vec_splat_f32(s[1]));
            HVX_سمتیہ vscale2 = Q6_V_vmux_QVV(q64, hvx_vec_splat_f32(s[2]), hvx_vec_splat_f32(s[3]));
            HVX_سمتیہ vscale3 = Q6_V_vmux_QVV(q64, hvx_vec_splat_f32(s[4]), hvx_vec_splat_f32(s[5]));
            HVX_سمتیہ vscale4 = Q6_V_vmux_QVV(q64, hvx_vec_splat_f32(s[6]), hvx_vec_splat_f32(s[7]));

            HVX_سمتیہ v_ql0 = *(ثابت HVX_UVector *) (ql);
            HVX_سمتیہ v_ql1 = *(ثابت HVX_UVector *) (ql + 32);
            HVX_سمتیہ v_qh  = *(ثابت HVX_UVector *) (qh);

            HVX_سمتیہ ql0_lo = Q6_V_vand_VV(v_ql0, نقاب_0f);
            HVX_سمتیہ qh_0   = Q6_Vw_vasl_VwR(Q6_V_vand_VV(v_qh, نقاب_03), 4);
            HVX_سمتیہ v_q1   = Q6_Vb_vsub_VbVb(Q6_V_vor_VV(ql0_lo, qh_0), v32_b);

            HVX_سمتیہ ql1_lo = Q6_V_vand_VV(v_ql1, نقاب_0f);
            HVX_سمتیہ qh_1   = Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_qh, 2), نقاب_03), 4);
            HVX_سمتیہ v_q2   = Q6_Vb_vsub_VbVb(Q6_V_vor_VV(ql1_lo, qh_1), v32_b);

            HVX_سمتیہ ql0_hi = Q6_Vub_vlsr_VubR(v_ql0, 4);
            HVX_سمتیہ qh_2   = Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_qh, 4), نقاب_03), 4);
            HVX_سمتیہ v_q3   = Q6_Vb_vsub_VbVb(Q6_V_vor_VV(ql0_hi, qh_2), v32_b);

            HVX_سمتیہ ql1_hi = Q6_Vub_vlsr_VubR(v_ql1, 4);
            HVX_سمتیہ qh_3   = Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_qh, 6), نقاب_03), 4);
            HVX_سمتیہ v_q4   = Q6_Vb_vsub_VbVb(Q6_V_vor_VV(ql1_hi, qh_3), v32_b);

            HVX_سمتیہ_Pair p16_1 = Q6_Wh_vunpack_Vb(v_q1);
            HVX_سمتیہ_Pair p32_1 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_1));
            HVX_سمتیہ     v_f32_1 = Q6_Vsf_equals_Vw(Q6_V_lo_W(p32_1));
            hvx_vmem(y + 0) = hvx_vec_mul_f32_f32(v_f32_1, vscale1);

            HVX_سمتیہ_Pair p16_2 = Q6_Wh_vunpack_Vb(v_q2);
            HVX_سمتیہ_Pair p32_2 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_2));
            HVX_سمتیہ     v_f32_2 = Q6_Vsf_equals_Vw(Q6_V_lo_W(p32_2));
            hvx_vmem(y + 32) = hvx_vec_mul_f32_f32(v_f32_2, vscale2);

            HVX_سمتیہ_Pair p16_3 = Q6_Wh_vunpack_Vb(v_q3);
            HVX_سمتیہ_Pair p32_3 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_3));
            HVX_سمتیہ     v_f32_3 = Q6_Vsf_equals_Vw(Q6_V_lo_W(p32_3));
            hvx_vmem(y + 64) = hvx_vec_mul_f32_f32(v_f32_3, vscale3);

            HVX_سمتیہ_Pair p16_4 = Q6_Wh_vunpack_Vb(v_q4);
            HVX_سمتیہ_Pair p32_4 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_4));
            HVX_سمتیہ     v_f32_4 = Q6_Vsf_equals_Vw(Q6_V_lo_W(p32_4));
            hvx_vmem(y + 96) = hvx_vec_mul_f32_f32(v_f32_4, vscale4);

            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// bytes row, row + 32, row + 64 and row + 96 of v, to bytes 0..3
ساکن اندرونی HVX_سمتیہ hvx_dequantize_gather4(HVX_سمتیہ v, uint32_t صف) {
    v = Q6_V_vror_VR(v, صف);
    HVX_سمتیہ packed = Q6_V_vmux_QVV(Q6_Q_vsetq_R(1), v, Q6_V_vror_VR(v, 31));
    packed = Q6_V_vmux_QVV(Q6_Q_vsetq_R(2), packed, Q6_V_vror_VR(v, 62));
    واپس Q6_V_vmux_QVV(Q6_Q_vsetq_R(3), packed, Q6_V_vror_VR(v, 93));
}

ساکن اندرونی خالی hvx_dequantize_صف_tiled_q8_0_f32(اعشاری * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t صف) {
    ثابت HVX_سمتیہ_Pred first2 = Q6_Q_vsetq_R(2);
    ثابت HVX_سمتیہ_Pred first4 = Q6_Q_vsetq_R(4);

    برائے (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ثابت uint8_t * tile = src_spad + k_tile * tile_stride;
        اعشاری * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_ستون_تعداد;

        HVX_سمتیہ vq = Q6_V_vzero();
        برائے (صحیح گروہ = 7; گروہ >= 0; --گروہ) {
            ثابت HVX_سمتیہ v = Q6_V_vror_VR(hvx_vmem(tile + گروہ * VLEN), 2 * صف);
            ثابت HVX_سمتیہ packed = Q6_V_vmux_QVV(first2, v, Q6_V_vror_VR(v, 62));
            vq = Q6_V_vmux_QVV(first4, packed, Q6_V_vror_VR(vq, VLEN - 4));
        }

        ثابت HVX_سمتیہ scales = hvx_vmem(tile + 1024);
        ثابت HVX_سمتیہ scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 2 * صف));
        ثابت HVX_سمتیہ scale = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(scale_hf));
        ثابت HVX_سمتیہ_Pair p16 = Q6_Wh_vunpack_Vb(vq);
        ثابت HVX_سمتیہ_Pair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        HVX_سمتیہ اقدار = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
        *(HVX_سمتیہ *) dst = اقدار;
    }
}

ساکن اندرونی خالی hvx_dequantize_صف_tiled_q4_0_f32(اعشاری * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t صف) {
    ثابت HVX_سمتیہ_Pred first4 = Q6_Q_vsetq_R(4);

    برائے (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ثابت uint8_t * tile = src_spad + k_tile * tile_stride;
        اعشاری * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_ستون_تعداد;

        HVX_سمتیہ vq = Q6_V_vzero();
        برائے (صحیح گروہ = 3; گروہ >= 0; --گروہ) {
            vq = Q6_V_vmux_QVV(first4, hvx_dequantize_gather4(hvx_vmem(tile + گروہ * VLEN), صف), Q6_V_vror_VR(vq, VLEN - 4));
        }

        ثابت HVX_سمتیہ lo = Q6_V_vand_VV(vq, Q6_Vb_vsplat_R(0x0F));
        ثابت HVX_سمتیہ hi = Q6_Vub_vlsr_VubR(vq, 4);
        vq = Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -1));
        vq = Q6_Vb_vsub_VbVb(vq, Q6_Vb_vsplat_R(8));

        ثابت HVX_سمتیہ scales = hvx_vmem(tile + 512);
        ثابت HVX_سمتیہ scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 2 * صف));
        ثابت HVX_سمتیہ scale = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(scale_hf));
        ثابت HVX_سمتیہ_Pair p16 = Q6_Wh_vunpack_Vb(vq);
        ثابت HVX_سمتیہ_Pair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        HVX_سمتیہ اقدار = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
        *(HVX_سمتیہ *) dst = اقدار;
    }
}

ساکن اندرونی خالی hvx_dequantize_صف_tiled_q4_k_f32(اعشاری * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t صف) {
    ثابت HVX_سمتیہ_Pred first4 = Q6_Q_vsetq_R(4);

    برائے (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ثابت uint8_t * tile = src_spad + k_tile * tile_stride;
        اعشاری * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_ستون_تعداد;

        HVX_سمتیہ vq = Q6_V_vzero();
        برائے (صحیح گروہ = 3; گروہ >= 0; --گروہ) {
            vq = Q6_V_vmux_QVV(first4, hvx_dequantize_gather4(hvx_vmem(tile + گروہ * VLEN), صف), Q6_V_vror_VR(vq, VLEN - 4));
        }

        ثابت HVX_سمتیہ lo = Q6_V_vand_VV(vq, Q6_Vb_vsplat_R(0x0F));
        ثابت HVX_سمتیہ hi = Q6_Vub_vlsr_VubR(vq, 4);
        vq = Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -1));

        ثابت HVX_سمتیہ scales = hvx_vmem(tile + 512);
        ثابت HVX_سمتیہ scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 4 * صف));
        ثابت HVX_سمتیہ scale = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(scale_hf));
        ثابت HVX_سمتیہ انحراف_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 4 * صف + 2));
        ثابت HVX_سمتیہ انحراف = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(انحراف_hf));

        ثابت HVX_سمتیہ_Pair p16 = Q6_Wh_vunpack_Vb(vq);
        ثابت HVX_سمتیہ_Pair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        HVX_سمتیہ اقدار = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
        *(HVX_سمتیہ *) dst = hvx_vec_جوڑو_f32_f32(اقدار, انحراف);
    }
}

ساکن اندرونی خالی hvx_dequantize_صف_tiled_q5_k_f32(اعشاری * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t صف) {
    ثابت HVX_سمتیہ_Pred first4 = Q6_Q_vsetq_R(4);
    ثابت HVX_سمتیہ one = Q6_Vb_vsplat_R(1);

    برائے (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ثابت uint8_t * tile = src_spad + k_tile * tile_stride;
        اعشاری * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_ستون_تعداد;

        ثابت HVX_سمتیہ p5 = hvx_dequantize_gather4(hvx_vmem(tile + 640), صف);
        HVX_سمتیہ vq = Q6_V_vzero();
        HVX_سمتیہ vh = Q6_V_vzero();
        برائے (صحیح گروہ = 3; گروہ >= 0; --گروہ) {
            vq = Q6_V_vmux_QVV(first4, hvx_dequantize_gather4(hvx_vmem(tile + گروہ * VLEN), صف), Q6_V_vror_VR(vq, VLEN - 4));
            vh = Q6_V_vmux_QVV(first4, Q6_Vub_vlsr_VubR(p5, 2 * گروہ), Q6_V_vror_VR(vh, VLEN - 4));
        }

        ثابت HVX_سمتیہ lo = Q6_V_vor_VV(Q6_V_vand_VV(vq, Q6_Vb_vsplat_R(0x0F)), Q6_Vw_vasl_VwR(Q6_V_vand_VV(vh, one), 4));
        ثابت HVX_سمتیہ hi = Q6_V_vor_VV(Q6_Vub_vlsr_VubR(vq, 4), Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(vh, 1), one), 4));
        vq = Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -1));

        ثابت HVX_سمتیہ scales = hvx_vmem(tile + 512);
        ثابت HVX_سمتیہ scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 4 * صف));
        ثابت HVX_سمتیہ scale = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(scale_hf));
        ثابت HVX_سمتیہ انحراف_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 4 * صف + 2));
        ثابت HVX_سمتیہ انحراف = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(انحراف_hf));

        ثابت HVX_سمتیہ_Pair p16 = Q6_Wh_vunpack_Vb(vq);
        ثابت HVX_سمتیہ_Pair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        HVX_سمتیہ اقدار = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
        *(HVX_سمتیہ *) dst = hvx_vec_جوڑو_f32_f32(اقدار, انحراف);
    }
}

ساکن اندرونی خالی hvx_dequantize_صف_tiled_q6_k_f32(اعشاری * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t صف) {
    ثابت HVX_سمتیہ_Pred first4 = Q6_Q_vsetq_R(4);
    ثابت HVX_سمتیہ_Pred first16 = Q6_Q_vsetq_R(16 * حجم(اعشاری));
    ثابت HVX_سمتیہ نقاب_0f = Q6_Vb_vsplat_R(0x0F);
    ثابت HVX_سمتیہ نقاب_03 = Q6_Vb_vsplat_R(0x03);

    برائے (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ثابت uint8_t * tile = src_spad + k_tile * tile_stride;
        اعشاری * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_ستون_تعداد;

        HVX_سمتیہ vq = Q6_V_vzero();
        برائے (صحیح گروہ = 7; گروہ >= 0; --گروہ) {
            ثابت HVX_سمتیہ lo_plane = Q6_V_vror_VR(hvx_vmem(tile + (گروہ >> 1) * VLEN), 4 * صف);
            ثابت HVX_سمتیہ hi_plane = Q6_V_vror_VR(hvx_vmem(tile + 512 + (گروہ >> 2) * VLEN), 4 * صف);
            ثابت HVX_سمتیہ lo = (گروہ & 1) ? Q6_Vub_vlsr_VubR(lo_plane, 4) : Q6_V_vand_VV(lo_plane, نقاب_0f);
            ثابت HVX_سمتیہ hi = Q6_Vub_vlsr_VubR(hi_plane, 2 * (گروہ & 3));
            ثابت HVX_سمتیہ packed = Q6_V_vor_VV(lo, Q6_Vw_vasl_VwR(Q6_V_vand_VV(hi, نقاب_03), 4));
            vq = Q6_V_vmux_QVV(first4, packed, Q6_V_vror_VR(vq, VLEN - 4));
        }

        ثابت HVX_سمتیہ scales = hvx_vmem(tile + 768);
        ثابت HVX_سمتیہ scale_lo_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 2 * صف));
        ثابت HVX_سمتیہ scale_hi_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 64 + 2 * صف));
        ثابت HVX_سمتیہ scale_lo = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(scale_lo_hf));
        ثابت HVX_سمتیہ scale_hi = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(scale_hi_hf));
        ثابت HVX_سمتیہ scale = Q6_V_vmux_QVV(first16, scale_lo, scale_hi);
        ثابت HVX_سمتیہ_Pair p16 = Q6_Wh_vunpack_Vb(Q6_Vb_vsub_VbVb(vq, Q6_Vb_vsplat_R(32)));
        ثابت HVX_سمتیہ_Pair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        *(HVX_سمتیہ *) dst = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
    }
}

ساکن اندرونی HVX_سمتیہ hvx_dequantize_tiled_f16_gather4(ثابت uint8_t * p, uint32_t r_byte) {
    HVX_سمتیہ w0  = Q6_V_vror_VR(hvx_vmem(p + 0 * VLEN), r_byte);
    HVX_سمتیہ w1  = Q6_V_vror_VR(hvx_vmem(p + 1 * VLEN), r_byte);
    HVX_سمتیہ w01 = Q6_V_lo_W(Q6_W_vshuff_VVR(w1, w0, 4));
    HVX_سمتیہ w2  = Q6_V_vror_VR(hvx_vmem(p + 2 * VLEN), r_byte);
    HVX_سمتیہ w3  = Q6_V_vror_VR(hvx_vmem(p + 3 * VLEN), r_byte);
    HVX_سمتیہ w23 = Q6_V_lo_W(Q6_W_vshuff_VVR(w3, w2, 4));
    واپس Q6_V_lo_W(Q6_W_vshuff_VVR(w23, w01, 8));
}

ساکن اندرونی خالی hvx_dequantize_صف_tiled_f16_f32(اعشاری * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t صف) {
    ثابت uint32_t r_byte = 4 * صف;

    برائے (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ثابت uint8_t * tile = src_spad + k_tile * tile_stride;
        اعشاری * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_ستون_تعداد;

        HVX_سمتیہ g0 = hvx_dequantize_tiled_f16_gather4(tile + 0 * VLEN, r_byte);
        HVX_سمتیہ g1 = hvx_dequantize_tiled_f16_gather4(tile + 4 * VLEN, r_byte);
        HVX_سمتیہ g01 = Q6_V_lo_W(Q6_W_vshuff_VVR(g1, g0, 16));

        HVX_سمتیہ g2 = hvx_dequantize_tiled_f16_gather4(tile + 8 * VLEN, r_byte);
        HVX_سمتیہ g3 = hvx_dequantize_tiled_f16_gather4(tile + 12 * VLEN, r_byte);
        HVX_سمتیہ g23 = Q6_V_lo_W(Q6_W_vshuff_VVR(g3, g2, 16));

        HVX_سمتیہ g = Q6_V_lo_W(Q6_W_vshuff_VVR(g23, g01, 32));
        *(HVX_سمتیہ *) dst = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(g));
    }
}

ساکن اندرونی خالی hvx_dequantize_صف_tiled_q1_0_f32(اعشاری * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t صف) {
    ساکن ثابت uint32_t __attribute__((aligned(128))) ثنائی_ہندسہ_نقاب[32] = {
        1u <<  0, 1u <<  8, 1u << 16, 1u << 24,
        1u <<  1, 1u <<  9, 1u << 17, 1u << 25,
        1u <<  2, 1u << 10, 1u << 18, 1u << 26,
        1u <<  3, 1u << 11, 1u << 19, 1u << 27,
        1u <<  4, 1u << 12, 1u << 20, 1u << 28,
        1u <<  5, 1u << 13, 1u << 21, 1u << 29,
        1u <<  6, 1u << 14, 1u << 22, 1u << 30,
        1u <<  7, 1u << 15, 1u << 23, 1u << 31,
    };
    ثابت HVX_سمتیہ v_ثنائی_ہندسہ_نقاب = *(ثابت HVX_سمتیہ *) ثنائی_ہندسہ_نقاب;
    ثابت HVX_سمتیہ v_zero     = Q6_V_vzero();
    ثابت HVX_سمتیہ v_sign_ثنائی_ہندسہ = Q6_V_vsplat_R(0x80000000);

    برائے (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ثابت uint8_t * tile = src_spad + k_tile * tile_stride;
        اعشاری * dst = dst_spad + k_tile * HTP_TENSOR_TILE_K_Q1_0;

        ثابت HVX_سمتیہ scales   = hvx_vmem(tile + 512);
        ثابت HVX_سمتیہ scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 2 * صف));
        ثابت HVX_سمتیہ vd       = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(scale_hf));
        ثابت HVX_سمتیہ neg_vd   = Q6_V_vxor_VV(vd, v_sign_ثنائی_ہندسہ);

        #ہدایت unroll
        برائے (صحیح q = 0; q < 4; ++q) {
            ثابت HVX_سمتیہ v_بٹ = hvx_vec_repl_u32(Q6_V_vror_VR(hvx_vmem(tile + q * VLEN), 4 * صف));
            ثابت HVX_سمتیہ_Pred q_zero = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(v_بٹ, v_ثنائی_ہندسہ_نقاب), v_zero);
            hvx_vmem(dst + q * 32) = Q6_V_vmux_QVV(q_zero, neg_vd, vd);
        }
    }
}

ساکن اندرونی خالی hvx_dequantize_صف_tiled_q2_0_f32(اعشاری * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t صف) {
    ثابت HVX_سمتیہ نقاب_03 = Q6_Vb_vsplat_R(0x03);
    ثابت HVX_سمتیہ v_one   = Q6_Vb_vsplat_R(1);

    برائے (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        ثابت uint8_t * tile = src_spad + k_tile * tile_stride;
        اعشاری * dst = dst_spad + k_tile * HTP_TENSOR_TILE_K_Q2_0;

        ثابت HVX_سمتیہ v_packed = hvx_dequantize_tiled_f16_gather4(tile, 4 * صف);
        ثابت HVX_سمتیہ p0 = Q6_V_vand_VV(v_packed, نقاب_03);
        ثابت HVX_سمتیہ p1 = Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_packed, 2), نقاب_03);
        ثابت HVX_سمتیہ p2 = Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_packed, 4), نقاب_03);
        ثابت HVX_سمتیہ p3 = Q6_Vub_vlsr_VubR(v_packed, 6);

        ثابت HVX_سمتیہ p01 = Q6_V_lo_W(Q6_W_vshuff_VVR(p1, p0, -4));
        ثابت HVX_سمتیہ p23 = Q6_V_lo_W(Q6_W_vshuff_VVR(p3, p2, -4));
        ثابت HVX_سمتیہ vq  = Q6_Vb_vsub_VbVb(Q6_V_lo_W(Q6_W_vshuff_VVR(p23, p01, -8)), v_one);

        ثابت HVX_سمتیہ scales   = hvx_vmem(tile + 512);
        ثابت HVX_سمتیہ scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 2 * صف));
        ثابت HVX_سمتیہ scale    = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(scale_hf));

        ثابت HVX_سمتیہ_Pair p16 = Q6_Wh_vunpack_Vb(vq);
        ثابت HVX_سمتیہ_Pair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        hvx_vmem(dst + 0)  = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
        hvx_vmem(dst + 32) = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_hi_W(p32)), scale);
    }
}

#اگر_ختم // HVX_QUANT_H
